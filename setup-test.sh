mkdir -p data/books
cd data/books

# Classic high-quality public-domain books
wget https://www.gutenberg.org/files/1342/1342-0.txt
wget https://www.gutenberg.org/files/84/84-0.txt
wget https://www.gutenberg.org/files/11/11-0.txt
wget https://www.gutenberg.org/files/1661/1661-0.txt
wget https://www.gutenberg.org/files/98/98-0.txt
wget https://www.gutenberg.org/files/2701/2701-0.txt
wget https://www.gutenberg.org/files/74/74-0.txt
wget https://www.gutenberg.org/files/76/76-0.txt
wget https://www.gutenberg.org/files/345/345-0.txt
wget https://www.gutenberg.org/files/5200/5200-0.txt
wget https://www.gutenberg.org/files/4300/4300-0.txt
wget https://www.gutenberg.org/files/6130/6130-0.txt
wget https://www.gutenberg.org/files/174/174-0.txt
wget https://www.gutenberg.org/files/1400/1400-0.txt
wget https://www.gutenberg.org/files/2554/2554-0.txt

cd ../..

# Merge into one corpus
cat data/books/*.txt > data/corpus.txt

# Build larger tokenizer
./build/build_vocab data/corpus.txt 4096 data/tokenizer.vocab

# Prepare dataset
./build/prepare_data data/corpus.txt data/tokenizer.vocab

# Train
./build/train

# Generate
./build/generate checkpoints/ckpt_01500.bin data/tokenizer.vocab \
  --prompt "The transformer architecture" \
  --temp 0.8