# Stock splits test task

## Requirements
* Docker
* Make
* Unbound 8000 port

## Usage
To run the project, execute command from the root of the project 
```bash
make main
```
Starting may take up to 10 minutes, depending on the CPU power.  
After model is trained, web UI will be available at `127.0.0.1:8000`.  
You should upload csv file with some individual stock daily data for service to generate possible stock splits dates.  
Check CSV file [example](data/aapl.csv).