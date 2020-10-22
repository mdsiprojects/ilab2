#!/bin/bash
pdoc --html helpers -f  --template-dir ./templates

mkdir md
mkdir md/helpers
html2text.exe ./html/helpers/index.html > ./md/helpers/index.md
html2text.exe ./html/helpers/hansard.html > ./md/helpers/hansard.md
html2text.exe ./html/helpers/io.html > ./md/helpers/io.md
html2text.exe ./html/helpers/kld.html > ./md/helpers/kld.md
html2text.exe ./html/helpers/potus.html > ./md/helpers/potus.md
html2text.exe ./html/helpers/process.html > ./md/helpers/process.md
html2text.exe ./html/helpers/topics.html > ./md/helpers/topics.md
