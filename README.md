# Receptive_Field_Development
Developmental model of orientation selective receptive fields based on [[Miller 1994](https://www.jneurosci.org/content/14/1/409.short),[Fumarola 2021](https://arxiv.org/abs/2109.02048)]. Simulation of Hebbian plasticity in feedforward connections from a layer modeling thalamus to a layer modeling visual cortex. 

### Organization of the  project

The project has the following structure:

    Receptive_Field_Development/
      |- README.md
      |- tools/
         |- __init__.py
         |- network_system.py
         |- parse_args.py
         |- plotting.py
         |- rhs.py
         |- three_step_method.py
         |- tools.py
         |- data/
            |- ...
      |- __init__.py
      |- main.py

### Licensing

Licensed under MIT license.

### Getting cited

Code can be cited as

### Scripts
This repository contains a [Python Script](https://github.com/b3ttin4/Receptive_Field_Development/blob/main/main.py) that runs a simulation of a two-layer feedforward neural network. Run as

<code>python main.py --rI rI --rC rC</code>

where _rI_ gives intracortical interaction range and _rC_ gives spatial range of input correlations (called eta and zeta, respectively in [Fumarola 2021](https://arxiv.org/abs/2109.02048)).
