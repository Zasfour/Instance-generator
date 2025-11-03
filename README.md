# Instance Generator

This module generates academic instances for the urban drone trajectory optimization problem.

## Graph Structure

We consider a **grid-shaped directed graph**, where each edge has the same length (see the `GridSpec` function).  
The grid defines the spatial layout used to construct feasible horizontal paths for all flights.

## Common Parameters

To generate an instance, several parameters must be defined (see the `TimingParams` function), including:
- the maximum ground delay allowed before take-off,  
- the number of authorized flight levels,  
- the time required to ascent/descent at the nominal speed between two successive flight-levels,
- and the minimum and maximum cruise speeds.

These parameters are shared across all flights in the instance.

## Reference Flights

Each instance is built upon **five reference flights** (see the `BASE_FLIGHTS` object).  
Each reference flight is characterized by:
- its **relative departure time** with respect to the first reference flight (denoted `D1`), and  
- its **horizontal path** on the grid.

## Instance Generation

The generated instances contain $5 \times n$ flights, with \( n \in \mathbb{N}^* \).  
The reference set of five flights is replicated \( n-1 \) times.  
For each replication, the relative departure times are shifted by \( 60 \times (n-1) \) seconds (see the `generate_flight_intentions` function).

