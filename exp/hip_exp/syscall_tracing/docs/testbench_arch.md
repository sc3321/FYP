- One main testbench harness which is common to all programs. 
    Inside this harness do the common work for all variants and backends.
    Define both bytes and iterations and parse arguements.


- depending on backend, instantiate the backend object. Possibly do this as an IFDEF.

- Once a backend object instantiated, use the object to call into the backend program variation, i.e cuda alloc or hip kernels or something using the bytes and iterations initialised. 

- if warmup is set as an arguement then use that to have a trial start run. That can be varied based on backend


