GENERATE=src/create_model.py
MAIN=src/main.py
MODEL=Model.sav
IN=python

generate:
	$(IN) $(GENERATE)

run:
	$(IN) $(MAIN)

clean:
	rm $(MODEL)