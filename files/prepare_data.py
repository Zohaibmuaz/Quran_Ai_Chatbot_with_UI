import pandas as pd

def load_quran_file(filepath):
    data = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    parts = line.split('|')
                    if len(parts) < 3:
                        # print(f"Warning: Skipping malformed line {line_num} in {filepath}: '{line}'")
                        continue
                    key = f"{parts[0]}|{parts[1]}"
                    text = '|'.join(parts[2:])
                    data[key] = text
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    return data

print("Loading all source files (including English)...")
arabic_text = load_quran_file('quran-simple.txt')
maududi_translation = load_quran_file('ur.maududi.txt')
qadri_translation = load_quran_file('ur.qadri.txt')
english_translation = load_quran_file('en.maududi.txt') # <-- نئی فائل شامل کی گئی

if not all([arabic_text, maududi_translation, qadri_translation, english_translation]):
    print("One or more source files are missing. Exiting.")
    exit()

prepared_data = []
print("Merging all data to create a multilingual dataset...")
for key, arabic in arabic_text.items():
    entry = {
        "reference": key,
        "arabic": arabic,
        "translation_maududi": maududi_translation.get(key, ""),
        "translation_qadri": qadri_translation.get(key, ""),
        "translation_english": english_translation.get(key, "") # <-- نیا کالم شامل کیا گیا
    }
    prepared_data.append(entry)

df = pd.DataFrame(prepared_data)
output_filename = 'quran_multilingual_data.csv' # <-- فائل کا نیا نام
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\nSuccess! Your final multilingual data file '{output_filename}' has been created.")