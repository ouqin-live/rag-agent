# 知识库模块架构说明与优化方向

## 1. 当前架构

知识库模块位于 `rag_agent/knowledge/`，负责文档接入、分块、向量化和语义检索。

```
source (file/url)
   │
   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  BaseLoader  │ ──► │ BaseChunker  │ ──► │ BaseEmbedder │
│ (文档加载)    │     │  (文本分块)   │     │  (向量化)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           ┌──────┴──────┐
                                           │ VectorStore │
                                           │ (向量存储)   │
                                           └─────────────┘
```

## 2. 各组件职责与对应文件

### 2.1 抽象层 `base.py`

定义核心数据结构与接口：

- `Document`：源文档（id, content, source, metadata）
- `Chunk`：文档分块（id, text, doc_id, metadata, embedding）
- `RetrievalResult`：检索结果（chunk + score），提供 `text` 属性快捷访问
- `BaseLoader` / `BaseChunker` / `VectorStore`：三个抽象接口，便于替换实现

### 2.2 文档加载器 `loader.py`

支持多格式文档加载，根据 source 自动选择：

| 加载器 | 源类型 | 说明 |
|---|---|---|
| `TextLoader` | `.txt` | 读取纯文本文件 |
| `MarkdownLoader` | `.md` | 读取 markdown，去除 YAML frontmatter |
| `PdfLoader` | `.pdf` | 使用 PyMuPDF 提取文本 |
| `UrlLoader` | `http/https` | 下载网页，基础 HTML 转文本 |
| `AutoLoader` | 自动 | 根据后缀/URL 自动派发 |

文档 ID 由 `sha256(source + content)` 生成，保证相同内容有稳定 ID。

### 2.3 分块器 `chunker.py`

把长文档切分成适合检索的小块：

- `FixedSizeChunker`：按 `chunk_size` 切分，优先在句子边界断开，支持 `overlap`
- `RecursiveChunker`：按 段落 → 句子 → 词语 → 字符 递归切分

Chunk ID 由 `sha256(doc_id + index + text)` 决定，相同输入产生相同 ID，确保幂等性。

### 2.4 向量存储

当前默认使用 `ChromaVectorStore`（`chroma_store.py`），基于 Chroma 的 HNSW 索引，自动持久化。保留 `LocalVectorStore`（`store.py`，SQLite + numpy）作为兼容方案。

| 方案 | 文件 | 适用 |
|---|---|---|
| ChromaVectorStore | `chroma_store.py` | 默认推荐，HNSW 索引，支持 upsert |
| LocalVectorStore | `store.py` | 兼容保留，SQLite + numpy，零额外依赖 |

两者均实现 `VectorStore` 接口，`KnowledgeBase` 通过工厂方法切换：

```python
kb = KnowledgeBase.from_chroma_store("data/kb")   # 推荐
kb = KnowledgeBase.from_local_store("data/kb")     # 兼容
```

### 2.5 Embedding `embedder.py`

- `SentenceTransformerEmbedder`：加载 `BAAI/bge-small-zh-v1.5`（512 维）
- `FallbackEmbedding`：离线降级，基于字符随机投影
- 关键设计：fallback 维度自动与真实模型对齐

### 2.6 编排入口 `kb.py`

`KnowledgeBase` 串联整个流程：

- `add_document(source)`：加载 → 分块 → 向量化 → 增量更新 → 持久化
- `remove_document(doc_id)`：删除文档所有 chunks
- `search(query, top_k, filters)`：向量化查询 → 元数据过滤 → 语义检索

## 3. 核心数据流

### 3.1 文档入库

```
source (file/url)
   │
   ▼
loader.load() → Document[]
   │
   ▼
chunker.chunk(doc) → Chunk[]
   │
   ▼
embedder.encode(texts) → embeddings
   │
   ▼
store.delete_by_doc(doc_id)  ← 增量更新：删旧
store.add(chunks)            ← 写入新 chunks
store.persist()              ← 持久化
```

### 3.2 语义检索

```
query string
   │
   ▼
embedder.encode(query) → query_embedding
   │
   ▼
store.search(query_embedding, top_k, filters) → RetrievalResult[]
```

## 4. 当前局限

### 4.1 检索方式单一

仅支持 Dense 向量检索，缺少关键词（BM25）召回。纯向量检索对专有名词、数字、缩写不敏感。

### 4.2 无重排序

检索结果直接按相似度排序，没有重排序模型提升精度。

### 4.3 分块策略简单

固定长度或递归切分，不理解文档语义结构（标题层级、段落主题），可能导致 chunk 语义不完整。

### 4.4 URL 解析粗糙

`UrlLoader` 使用基础正则清洗 HTML，无法处理复杂网页结构、动态内容或 PDF 链接。

### 4.5 不支持多模态

仅处理文本，无法解析图片中的文字（OCR）、表格数据或图表。

### 4.6 增量更新粒度粗

当前是「删旧 → 插新」整文档替换，不对比差异。大文档修改几个字也会重建全部 chunk。

## 5. 可优化方向

### 5.1 混合检索（BM25 + Dense）

结合关键词检索和向量检索，互补长短：

```python
class HybridRetriever:
    def __init__(self, dense_store, bm25_index):
        ...

    def search(self, query, top_k):
        dense_results = self.dense_store.search(query)
        bm25_results = self.bm25_index.search(query)
        return self._reciprocal_rank_fusion(dense_results, bm25_results)
```

- Dense：语义相似，对同义词鲁棒
- BM25：精确匹配，对专有名词、数字敏感
- 通过 RRF 融合，NDCG 通常提升 10-15%

### 5.2 重排序模型

在粗排后加一个 Cross-Encoder 精排：

```
query + chunk → Cross-Encoder → relevance_score
```

可选模型：`BAAI/bge-reranker-base` / `cross-encoder/ms-marco-MiniLM-L-6-v2`

### 5.3 语义分块

按文档实际结构切分，保持段落完整性：

- 按 Markdown 标题层级分块
- 按段落主题相似度聚合（相邻句向量相似度陡降处切分）
- 保留父级标题作为 chunk 的前缀上下文

### 5.4 URL 加载增强

引入 `playwright` / `selenium` 处理动态网页，或接入 `trafilatura` 做专业文本提取。

### 5.5 多模态支持

- 图片 → OCR（pytesseract）/ 多模态 embedding
- PDF 表格 → `pymupdf` 表格提取
- 图表说明 → LLM 描述生成

### 5.6 分块策略可配置化

让 `KnowledgeBase` 支持运行时切换分块策略，而不需要重新实例化：

```python
kb = KnowledgeBase.from_chroma_store("data/kb")
kb.add_document("report.txt", chunker=SemanticChunker())
kb.add_document("code.py", chunker=RecursiveChunker(chunk_size=200))
```

### 5.7 增量更新优化

只更新有差异的 chunk，不重建全文档：

```python
def smart_update(new_chunks):
    for new_chunk in new_chunks:
        old = store.get(new_chunk.id)
        if old and old.text == new_chunk.text:
            continue
        store.upsert([new_chunk])
```

## 6. 优先级建议

| 优化方向 | 优先级 | 说明 |
|---|---|---|
| 混合检索（BM25 + Dense） | P0 | 对检索质量提升最大，是 RAG 核心能力 |
| 重排序模型 | P1 | 在混合检索基础上进一步提升精度 |
| 语义分块 | P1 | 改善 chunk 质量，直接影响检索和生成效果 |
| 分块策略可配置化 | P2 | 降低运维复杂度 |
| URL 加载增强 | P2 | 网络场景的文档接入体验 |
| 增量更新优化 | P3 | 大文档场景性能优化 |
| 多模态支持 | P3 | 扩展应用场景 |
