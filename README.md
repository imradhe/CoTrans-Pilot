# ✅ IndicVoices Linguistic Studies — Final Checklist

## 1. **Data Loading**
- Load dataset from `.csv` or `.parquet`.

## 2. **Data Cleaning**
- **Deduplication:** Remove duplicate rows based on the `verbatim` column.

## 3. **Retain Specific Columns Only**
Keep only the following columns:

```python
['id', 'lang', 'verbatim', 'normalized', 'scenario', 'gender', 'age_group', 
 'job_type', 'qualification', 'area', 'district', 'state', 'occupation', 
 'unsanitized_verbatim', 'unsanitized_normalized']
```

## 4. **Utterance Statistics (per Language and Scenario)**
- For each language, plot 4 bars:
  - `total`, `read`, `extempore`, `conversation`

## 5. **Define Noise Tags and Disfluency Tags**
- **Noise tags:**  
  
  ```python
  noise_tags = ["TV", "animal", "baby", "baby_crying", "baby_talking", "barking", "beep", "bell",
    "bird_squawk", "breathing", "buzz", "buzzer", "child", "child_crying", "child_laughing",
    "child_talking", "child_whining", "child_yelling", "children", "children_talking",
    "children_yelling", "chiming", "clanging", "clanking", "click", "clicking", "clink",
    "clinking", "cough", "dishes", "door", "footsteps", "gasp", "groan", "hiss", "hmm",
    "horn", "hum", "inhaling", "laughter", "meow", "motorcycle", "music", "noise",
    "nose_blowing", "Persistent-noise-end", "Persistent-noise-start", "phone_ringing",
    "phone_vibrating", "popping", "pounding", "printer", "rattling", "ringing", "rustling",
    "scratching", "screeching", "sigh", "singing", "siren", "smack", "sneezing", "sniffing",
    "Sniffle", "snorting", "squawking", "squeak", "stammers", "static", "swallowing",
    "talking", "tapping", "throat_clearing", "thumping", "tone", "tones", "trill", "tsk",
    "typewriter", "ugh", "uhh", "uh-huh", "umm", "unintelligible", "wheezing", "whispering",
    "whistling", "yawning", "yelling"]
    ```

- **Disfluency tags (subset):**  

    ```python

    disfluency_tags = ["cough", "gasp", "groan", "hiss", "hmm", "hum", "inhaling", "laughter", "sigh",
    "sneezing", "sniffing", "Sniffle", "snorting", "stammers", "swallowing", "throat_clearing",
    "tsk", "ugh", "uhh", "uh-huh", "umm", "wheezing", "whispering", "yawning"]

    ```

## 6. **Code-Mixing Classification**
- For each utterance, check `unsanitized_normalized`
- Ignore noise/disfluency tags
- If English words are present → label as `code-mixed`, else `standard`
- Add column: `type = [code-mixed | standard]`

## 7. **Code-Mix Statistics (per Language and Scenario)**
- For each language (as separate plot), plot 8 bars:
  - `total` → `[code-mixed, standard]`
  - `read` → `[code-mixed, standard]`
  - `extempore` → `[code-mixed, standard]`
  - `conversation` → `[code-mixed, standard]`

## 8. **Vocabulary Statistics**
- Total vocabulary from `verbatim` + `normalized`
- Vocabulary size per language
- Vocabulary set per district within each language

## 9. **Exclusive vs Shared Vocabulary (per District)**
- For each language, plot:
  - Exclusive vocabulary vs shared vocabulary across districts

## 10. **Disfluency Density Metric**
- For each utterance, compute:

  ```
  disfluency_density = (# disfluency tags) / (# total words excluding tags)
  ```

- Add column: `disfluency_density`

## 11. **Disfluency Density Plots**
- Per language across **districts**
- Per language across **scenarios**
- Per language comparing **code-mixed vs standard**

## 12. **WER Calculation**
- For each utterance, compute **WER** between `verbatim` and `normalized`
- Add column: `wer`

## 13. **WER Plots**
- Average **WER per language**
- Average **WER per language across all districts**