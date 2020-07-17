extract_dir := Data

.PHONY: all
all: unzip

.PHONY: unzip
unzip:
	for i in *.zip; do unzip -o "$$i" -d $(extract_dir); done

.PHONY: clean
clean:
	rm -rf $(extract_dir)
