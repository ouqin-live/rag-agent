# 文档解析现状分析与改造方案

## 一、当前能力盘点

### 1.1 已支持的格式（loader.py）

- .txt：直接读 UTF-8 文本，零处理
- .md / .markdown：读文本 + 去 YAML frontmatter
- .pdf：PyMuPDF 提取文本 + 表格 + 元信息
- .docx：python-docx 提取文本 + 表格 + 标题层级
- .html / .htm：支持本地 HTML 文件，trafilatura 优先提取
- http/https URL：trafilatura 优先提取文本，降级正则，支持 PDF URL

### 1.2 分块能力（chunker.py）

- FixedSizeChunker：按固定大小切，优先句子边界，带 overlap
- RecursiveChunker：段落 → 句子 → 词语 → 字符，递归降级切
- SemanticChunker：用 embedding 算相邻句相似度，断点处切
- MarkdownStructureChunker：按标题层级切分，带父标题上下文，代码块/表格保护

四种分块器覆盖了从简单到智能的完整需求。MarkdownStructureChunker 补上了结构感知的短板。

### 1.3 关键定性

当前是「能跑」的级别——能处理最基础的 txt/md/pdf/网页，但解析质量粗糙，格式覆盖严重不足。文档检索的质量上限被解析环节卡死了。

---

## 二、具体缺口拆解

### 2.1 缺失的文档格式（完全没支持）

- ~~Word：.docx / .doc~~ ✅ 已实现 DocxLoader
- Excel：.xlsx / .csv — 结构化数据零支持
- PPT：.pptx — 演示文稿无法处理
- ~~HTML 本地文件：.html~~ ✅ 已实现 HtmlLoader
- ePub：电子书格式不支持
- 代码文件：没有专门 loader，落到 TextLoader 纯文本读

### 2.2 PDF 解析太浅

现在做的事：读文本 + 表格提取 + 元信息提取

没做的事：
- ~~表格提取~~ ✅ 已启用 PyMuPDF page.find_tables()
- ~~文档元信息（标题、作者、日期）~~ ✅ 已提取 doc.metadata
- 图片里的文字（OCR）
- 多栏排版识别（双栏 PDF 读出来顺序全乱）
- 目录 / 书签结构

### 2.3 URL 解析太粗糙

~~现在做的事：正则硬洗 HTML~~ ✅ 已升级：
- trafilatura 优先提取，质量大幅提升
- 正则方案保留作为降级 fallback
- PDF URL 支持：自动下载 → PdfLoader

仍缺：
- JS 渲染的内容完全拿不到

### 2.4 分块不理解文档结构

~~三段论：所有文档 → 纯文本 → 切块~~ ✅ 已实现 MarkdownStructureChunker：
- 按标题层级切分（h2 大段，h3 小段）
- 每个 chunk 带父标题上下文前缀
- 代码块和表格作为原子单元保护
- 过小 chunk 自动合并，超大 chunk 二次切分

仍缺：
- 非 Markdown 文档（PDF/Word）的结构感知分块

### 2.5 跨文档能力缺失

- 没有内容去重：同一份文档入库两次就是两份 chunk
- 没有元数据提取流水线：文件名之外的标题、作者、日期都要手工传
- 没有目录/文件夹递归导入：只能一个文件一个文件加

---

## 三、改造方案

### 3.1 阶段一：格式补齐（P0-P1）

目标：常用文档格式全覆盖

#### Word 加载器（DocxLoader）
- 依赖：python-docx
- 能力：提取文本 + 表格 + 段落样式（用样式区分标题/正文）
- 工作量：小（python-docx API 很简单）

#### Excel 加载器（ExcelLoader）
- 依赖：openpyxl（.xlsx）+ csv 内置（.csv）
- 能力：每个 sheet 转 Markdown 表格文本，保留行列结构
- 工作量：小

#### PPT 加载器（PptxLoader）
- 依赖：python-pptx
- 能力：提取每页幻灯片文本，保留标题/正文层级
- 工作量：小

#### HTML 本地文件支持
- 复用 UrlLoader 的提取逻辑，加本地 .html 文件读取
- 工作量：极小（改 AutoLoader 注册即可）

### 3.2 阶段二：解析质量升级（P1-P2）

目标：已支持的格式提解析质量

#### PDF 增强
- 启用 PyMuPDF 表格提取：page.find_tables()
- 文本提取时保留排版信息（textpage 的 block 结构）
- 提取文档元信息：doc.metadata

#### URL 解析升级
- 引入 trafilatura 做专业网页文本提取
- 保留现有正则方案做降级 fallback
- 支持 PDF URL（下载后走 PdfLoader）

#### Markdown 结构感知分块（MarkdownStructureChunker）
- 解析标题层级，按 h1/h2 切大段，h3 切小块
- 每个 chunk 带父标题作为上下文前缀
- 表格、代码块作为独立 chunk，不打散

### 3.3 阶段三：智能化（P3）

目标：用 AI 提升解析上限

#### OCR 能力
- 依赖：pytesseract + pdf2image（PDF 图片页）
- PDF 中图片先 OCR 再并回文本流

#### 文档结构理解
- 用 LLM 识别文档类型（报告/论文/合同/手册）自动调解析策略
- 表格语义化：LLM 把表格转自然语言描述

#### 批量导入
- 支持文件夹递归导入
- 自动提取文件名/目录名做 metadata
- 文件修改时间追踪，只重建变更文档

---

## 四、实施优先级

按投入产出比排序：

1. ~~Word 支持（DocxLoader）~~ ✅ 已完成
   企业场景最常用，python-docx 零门槛

2. ~~Markdown 结构感知分块~~ ✅ 已完成
   补上结构理解后检索精度提升明显

3. ~~URL 解析升级（trafilatura）~~ ✅ 已完成
   替换粗正则，质量提升立竿见影

4. ~~PDF 表格提取~~ ✅ 已完成
   PyMuPDF 原生支持，纯补漏

5. Excel / CSV 支持 — P2，半天
   理由：结构化数据场景需要，openpyxl 简单

6. PPT 支持 — P2，半天
   理由：演示文稿场景，python-pptx 简单

7. ~~HTML 本地文件~~ ✅ 已完成
   改 AutoLoader 注册即可

8. OCR / LLM 文档理解 — P3，按需
   理由：依赖重、场景特定，先做前面的快速见效项

---

## 五、不改什么（明确边界）

以下暂时不动，避免过度设计：

- 不做全文检索引擎（Elasticsearch 等），Chromadb + BM25 够用
- 不做视频/音频解析
- 不做多语言 OCR（中文 OCR 够用）
- 不做实时文档同步（文件夹 watch）
- 不做文档格式转换器（docx→md 之类）
