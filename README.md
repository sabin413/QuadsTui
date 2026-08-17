It's a TUI for QuADS. It is based on textual (https://textual.textualize.io), a Python API for terminal user interface development.
To use this TUI:
1. clone this repo
2. run "pixi run python quads_viewer.py" on your terminal.
Then, you need to give it two inputs it will ask for: model (such as geosfp) and date (such as 2024-02-01).

It will then open a graphical interface on your terminal. What you see in the display is mostly self explanatory. If you need to know the 
latitude stratum classification scheme, see the following: 

STRATA:

  Strat1:
    lat:
      min: -90
      max: -70

  Strat2:
    lat:
      min: -70
      max: -45

  Strat3:
    lat:
      min: -45
      max: -20

  Strat4:
     lat:
       min: -20
       max: 0

  Strat5:
    lat:
      min: 0
      max: 20

  Strat6:
    lat:
      min: 20
      max: 45

  Strat7:
    lat:
      min: 45
      max: 70

  Strat8:
    lat:
      min: 70
      max: 90
