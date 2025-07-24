FROM axolotlai/axolotl:0.11.0

RUN pip install tensorkube pyyaml 

COPY axolotl-train.py .

COPY axolotl-config.yaml .

CMD ["python", "axolotl-train.py"]