# Cheat Sheet — Python File I/O & Pandas Basics
# 치트시트 — 파이썬 파일 입출력 & 판다스 기초

> Hands-on ML Chapter 2 prep. Companion to `getdata_and_pdbase.ipynb`.
> 핸즈온 ML 2장 준비용. `getdata_and_pdbase.ipynb` 정리본.

---

## Part 1. Python File I/O · 파이썬 파일 다루기

### 1-1. `open` and `with` · 파일 열기

```python
f = open("data.txt", "r")    # works, but you must call f.close() yourself
text = f.read()
f.close()                    # forget this → file stays open (leak)

with open("data.txt", "r", encoding="utf-8") as f:   # closes automatically
    text = f.read()
```

`with` closes the file even if an error happens inside the block.
/ `with` 블록은 안에서 에러가 나도 파일을 자동으로 닫아줌. 그냥 항상 `with` 써.

| Mode | English | 한국어 |
| --- | --- | --- |
| `"r"` | Read (default). Error if file missing | 읽기 (기본값). 파일 없으면 에러 |
| `"w"` | Write. **Erases** existing content | 쓰기. 기존 내용 **싹 지움** |
| `"a"` | Append to the end | 끝에 이어 쓰기 |
| `"x"` | Create new. Error if file exists | 새로 만들기. 이미 있으면 에러 |
| `"rb"` / `"wb"` | Binary read / write (images, archives) | 바이너리 읽기/쓰기 (이미지, 압축파일) |

| Code | English | 한국어 |
| --- | --- | --- |
| `f.read()` | Whole file as one string | 파일 전체를 문자열 하나로 |
| `f.readlines()` | List of lines (with `\n`) | 줄 단위 리스트 (`\n` 포함) |
| `for line in f:` | Loop line by line (memory-friendly) | 한 줄씩 순회 (메모리 아낌) |
| `line.strip()` | Remove `\n` and surrounding spaces | 줄바꿈·양끝 공백 제거 |
| `f.write("text\n")` | Write a string (add `\n` yourself) | 문자열 쓰기 (`\n`은 직접 붙여야 함) |

### 1-2. `pathlib` · 경로 다루기

```python
from pathlib import Path

p = Path("datasets") / "housing" / "housing.csv"   # join with /
p.is_file()                                         # exists and is a file?
p.exists()                                          # exists at all?
Path("datasets").mkdir(parents=True, exist_ok=True) # make folder, no error if it exists
p.name, p.suffix, p.parent                          # "housing.csv", ".csv", datasets/housing
```

`Path` objects work directly in `open()` and `pd.read_csv()`.
/ `Path` 객체는 `open()`, `pd.read_csv()`에 그대로 넣어도 됨.

### 1-3. `try` / `except` · 예외 처리

```python
try:
    df = pd.read_csv("missing.csv")
except FileNotFoundError:            # catch a SPECIFIC error
    print("file not found")
except (ValueError, KeyError) as e:  # several types; e holds the message
    print("bad data:", e)
else:
    print("runs only if NO error")
finally:
    print("runs ALWAYS (cleanup)")
```

| Keyword | English | 한국어 |
| --- | --- | --- |
| `try` | Code that might fail | 실패할 수도 있는 코드 |
| `except X` | Runs if error type X happens | X 에러가 나면 실행 |
| `else` | Runs if nothing failed | 에러 없을 때만 실행 |
| `finally` | Always runs | 무조건 실행 |
| `raise ValueError("msg")` | Throw an error yourself | 직접 에러 발생시키기 |

`pass` vs `continue` — they are **not** the same:
/ `pass`랑 `continue`는 **다름**:

```python
for name in ["a.csv", "b.csv", "c.csv"]:
    try:
        df = pd.read_csv(name)
    except FileNotFoundError:
        continue          # skip to the NEXT loop iteration / 다음 반복으로 넘어감
    print(name, df.shape) # skipped when continue fires

try:
    risky()
except ValueError:
    pass                  # do nothing, just go on / 아무것도 안 하고 그냥 진행
```

- `continue` only works **inside a loop**. / `continue`는 **반복문 안에서만** 됨.
- Avoid bare `except:` — it also swallows typos and Ctrl+C. Name the error type.
  / 그냥 `except:`는 오타·Ctrl+C까지 다 삼켜버림. 에러 타입 꼭 적어.

| Common error | When | 한국어 |
| --- | --- | --- |
| `FileNotFoundError` | Path doesn't exist | 경로에 파일 없음 |
| `KeyError` | Missing dict key / DataFrame column | 없는 딕셔너리 키 / 없는 열 이름 |
| `IndexError` | List index out of range | 리스트 범위 밖 인덱스 |
| `ValueError` | Right type, wrong value (`int("abc")`) | 타입은 맞는데 값이 이상함 |
| `TypeError` | Wrong type (`"a" + 1`) | 타입 자체가 안 맞음 |

### 1-4. Download + extract pattern · 다운로드·압축해제 패턴 (Ch.2)

```python
from pathlib import Path
import tarfile
import urllib.request
import pandas as pd

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():                      # download only once
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))   # OUTSIDE the if

housing = load_housing_data()
```

Watch the indentation of `return`: if it sits inside the `if`, the second run (file already there) returns `None`.
/ `return` 들여쓰기 주의: `if` 안에 있으면 두 번째 실행부터(파일 이미 있음) `None` 반환함.

---

## Part 2. Pandas · 판다스

### 2-1. Core idea · 핵심 구조

- **Series** = one labeled column (1-D). / 라벨 붙은 열 하나 (1차원).
- **DataFrame** = several Series sharing one index (2-D table). / 같은 인덱스를 공유하는 Series 묶음 (2차원 표).

```python
import pandas as pd

s  = pd.Series([1, 2, 3])
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})   # key = column name

df = pd.read_csv("file.csv")
df.to_csv("out.csv", index=False)   # index=False: don't write the 0,1,2… column
```

### 2-2. Inspect · 훑어보기

| Code | English | 한국어 |
| --- | --- | --- |
| `df.shape` | (rows, cols) — attribute, **no `()`** | (행, 열) — 속성이라 **괄호 없음** |
| `df.head(n)` / `df.tail(n)` | First / last n rows (default 5) | 앞 / 뒤 n행 (기본 5) |
| `df.info()` | Columns, dtypes, non-null counts | 열·타입·결측 아닌 개수 |
| `df.describe()` | Stats of numeric columns | 숫자 열 통계 요약 |
| `df.dtypes` | Type per column | 열별 타입 |
| `df["col"].value_counts()` | Count per unique value, sorted | 값별 개수, 많은 순 정렬 |
| `df.isna().sum()` | Missing values per column | 열별 결측치 개수 |

### 2-3. Select · 고르기

| Code | Returns | 한국어 |
| --- | --- | --- |
| `df["a"]` | Series | 열 하나 → Series |
| `df[["a", "b"]]` | DataFrame (**double brackets**) | 여러 열 → DataFrame (**대괄호 두 겹**) |
| `df.loc[1, "a"]` | Single value — row **label** 1, col "a" | 값 하나 — 행 **라벨** 1, 열 "a" |
| `df.iloc[0, 1]` | Single value — row **position** 0, col position 1 | 값 하나 — **위치** 0행, 1열 |
| `df.loc[:, "a":"b"]` | Label slice, **end included** | 라벨 슬라이스, **끝 포함** |
| `df.iloc[0:2, :]` | Position slice, **end excluded** | 위치 슬라이스, **끝 미포함** |

`df["a", "b"]` (single brackets, two names) is a **KeyError** — it looks for one column named `("a", "b")`.
/ `df["a", "b"]`처럼 한 겹 대괄호에 이름 두 개 넣으면 **KeyError**. `("a", "b")`라는 열 하나를 찾음.

### 2-4. Filter with boolean masks · 불리언 마스크로 거르기

```python
mask = df["a"] > 2          # Series of True/False  → 조건 결과 True/False Series
df[mask]                    # keep rows where True  → True인 행만 남김
df[df["a"] > 2]             # same thing in one line

df[(df["a"] > 1) & (df["b"] < 6)]   # AND → &   (parentheses required / 괄호 필수)
df[(df["a"] == 1) | (df["b"] == 6)] # OR  → |
df[~(df["a"] > 2)]                  # NOT → ~

df[df["city"].isin(["Berlin", "Erlangen"])]     # value in a list / 목록 안에 있는 값
df.query("age > 20 and city == 'Erlangen'")     # string form; here `and` is OK
```

Use `&` `|` `~`, not `and` `or` `not` — except inside `query()`.
/ `and`/`or`는 에러 남. `&` `|` `~` 써. (`query()` 문자열 안에서만 `and` 허용.)

### 2-5. Modify · 바꾸기

| Code | English | 한국어 |
| --- | --- | --- |
| `df["c"] = df["a"] * 2` | New column (vectorized) | 새 열 (벡터 연산, for문 불필요) |
| `df.loc[df["a"] > 2, "b"] = 0` | Set values where condition holds | 조건 맞는 곳 값 바꾸기 |
| `df.rename(columns={"a": "x"})` | Rename columns | 열 이름 바꾸기 |
| `df.drop(columns=["c"])` | Drop columns | 열 삭제 |
| `df.dropna()` / `df.fillna(0)` | Drop / fill missing values | 결측치 삭제 / 채우기 |
| `df.sort_values("a", ascending=False)` | Sort descending | 내림차순 정렬 |
| `df["a"].astype(float)` | Change dtype | 타입 변환 |

Most methods **return a new DataFrame** — reassign: `df = df.dropna()`.
/ 대부분 **새 DataFrame을 반환**함. 원본 바꾸려면 다시 대입.

### 2-6. Group · 그룹 집계

```python
df.groupby("ocean_proximity")["median_house_value"].mean()   # mean per group / 그룹별 평균
df.groupby("ocean_proximity").size()                         # rows per group / 그룹별 행 수
```

### 2-7. To NumPy & axis · 넘파이 변환과 axis

```python
arr = df.to_numpy()      # → numpy.ndarray, index & column names are gone
type(arr)                # <class 'numpy.ndarray'>
```

- `axis=0` → down the rows, result **per column** (`df.mean(axis=0)` = column means).
  / 세로 방향으로 계산 → 결과가 **열마다** 하나.
- `axis=1` → across the columns, result **per row** (`df.mean(axis=1)` = row means).
  / 가로 방향으로 계산 → 결과가 **행마다** 하나.

### 2-8. Debug reflex · 막힐 때 반사적으로

```python
type(x)     # Series? DataFrame? ndarray? → decides which methods exist
x.shape     # dimensions match what you expect?
x.dtype     # (Series/array) int, float, object(=string)?
```
