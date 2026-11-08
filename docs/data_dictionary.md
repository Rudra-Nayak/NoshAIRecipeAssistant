# Data Dictionary & Schema Specification

## 1. Source Datasets
The Eight Portions recipe dataset is composed of three raw JSON files located in `recipes_raw/`:

| Source Code | Source Provider | Filename | Raw Records | Usable Records | Usability Rate |
|---|---|---|---|---|---|
| `ar` | AllRecipes | `recipes_raw_nosource_ar.json` | 39,802 | 39,522 | 99.30% |
| `epi` | Epicurious | `recipes_raw_nosource_epi.json` | 25,323 | 25,261 | 99.76% |
| `fn` | Food Network | `recipes_raw_nosource_fn.json` | 60,039 | 58,155 | 96.86% |
| **Total** | | | **125,164** | **122,938** | **98.22%** |

---

## 2. Raw JSON Schema
Each file is a key-value JSON dictionary where:
- **Key**: 31-character unique alphanumeric identifier (e.g., `"rmK12Uau.ntP510KeImX506H6Mr6jTu"`).
- **Value**: A JSON Object detailing the recipe.

### Field Definitions

| Field Name | Type | Nullable | Description & Observations |
|---|---|---|---|
| `title` | `String` | Yes | Name of the dish. 280 nulls in `ar`, 289 nulls in `fn`. |
| `ingredients` | `List[String]` | Yes | Unstructured ingredient strings containing quantity, unit, preparation notes, and raw text. 1,629 empty lists in `fn`, 47 in `epi`. |
| `instructions` | `String` | Yes | Cooking steps formatted either as newline-separated blocks or continuous prose. 411 nulls in `fn`, 23 empty in `epi`. |
| `picture_link` | `String` | Yes | Legacy hashed asset pointer (e.g., `"55lznCYBbs2mT8BTx6BTkLhynGHzM.S"`). Unusable as public HTTP URLs; ignored during retrieval. |

---

## 3. Data Quality & Cleansing Rules

### 3.1 AllRecipes Advertisement Noise
In `recipes_raw_nosource_ar.json`, 39,518 recipes had raw ad-tracking tokens appended to ingredient lines:
- **Pattern A (Suffix)**: `"4 skinless, boneless chicken breast halves ADVERTISEMENT"`
- **Pattern B (Standalone)**: `"ADVERTISEMENT"`
- **Remediation**:
  1. Strip trailing `" ADVERTISEMENT"` substring from valid ingredient strings.
  2. Drop standalone `"ADVERTISEMENT"` entries from the ingredients list.

### 3.2 Global ID Uniqueness
- An analysis of all 125,164 keys across `ar`, `epi`, and `fn` revealed **0 duplicate IDs**.
- Every key in the raw dataset is globally unique, enabling direct usage as stable keys.

### 3.3 Usability Validation Rules
A record is deemed usable if and only if:
1. `title` is a non-null, non-empty string.
2. `ingredients` is a non-empty list containing at least one valid ingredient after cleaning.
3. `instructions` is a non-null, non-empty string.

Dropped records ($N=2,226$):
- `ar`: 280 records were empty JSON objects (`{}`).
- `epi`: 62 records had empty ingredients or empty instruction strings.
- `fn`: 1,884 records had null titles, empty ingredient lists, or missing instruction blocks.

---

## 4. Internal Unified Representation
Saved to `data/processed/recipes.pkl`:

```python
{
    "id": str,               # Globally unique 31-char key (e.g., 'rmK12Uau.ntP510KeImX506H6Mr6jTu')
    "source": str,           # 'ar' | 'epi' | 'fn'
    "title": str,            # Stripped original title
    "ingredients": list[str],# Cleaned ingredient strings (ADVERTISEMENT-free)
    "instructions": str,     # Original instructions
}
```

