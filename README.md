# Portfolio Backtester

Simple backtesting of portfolio with option to rebalance holdings yearly

## Description

A Jupyter Notebook that allows you to define a stock portfolio, assigning percentages to each holding.
Backtests the portfolio and compares with a benchmark.
Optionally re-balance the holdings yearly.

## Getting Started

### Dependencies

* Python 3.x
* Jupyter (jupyter-core: 4.6.1+ / jupyter-notebook: 6.0.3+ / jupyter-lab: 1.2.6)
    * pandas                    1.0.1+
    * pandas-datareader         0.8.1+


### Installing

* Clone this project locally

* With Anaconda
  * Install Anaconda 3.8+  (anaconda.org)
    * Install pandas-datareader. From the Terminal:
      * ```conda install pandas-datareader```

* Alternatively, without Anaconda
  * Install Python 3.x
  * ```python -m pip install --user pipenv``` # Install virtualenv
<!--
  * ```python --version```
       #Python 3.9.4
  * ```#python -m pip install --user jupyter matplotlib pandas pandas-datareader```
  * ```python -m pipenv install jupyter```
  * ```python -m pipenv install matplotlib pandas pandas-datareader```
  * ```jupyter notebook Backtest.ipynb``` # Run notebook
-->

### Executing program

* With Anaconda
  * Launch "Anaconda-Navigator"
  * Launch "JupyterLab" from the Anaconda Navigator front page.
  * Open ```Backtest.ipynb```
  * Run All Cells

* Without Anaconda
  * ```python -m pipenv run jupyter notebook Backtest.ipynb ``` # Activate virtualenv
  * Run All Cells
<!--
  * ```python -m pipenv shell``` # Activate virtualenv
-->

## Help

* Modify the portfolio and date range to customize
* If a stock is not avalable on Yahoo Finance, will get an exception

## Authors

* ajitn

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Uses Yahoo Finance to get historical data
