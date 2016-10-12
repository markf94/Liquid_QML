﻿namespace Microsoft.Research.Liquid

module UserSample =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    /// <summary>
    /// Performs an arbitrary rotation around X. 
    /// </summary>
    /// <param name="theta">Angle to rotate by</param>
    /// <param name="qs">The head qubit of this list is operated on.</param>
    let rotX (theta:float) (qs:Qubits) =
        let gate (theta:float) =
            let nam     = "Rx" + theta.ToString("F2")
            new Gate(
                Name    = nam,
                Help    = sprintf "Rotate in X by: %f" theta,
                Mat     = (
                    let phi     = theta / 2.0
                    let c       = Math.Cos phi
                    let s       = Math.Sin phi
                    CSMat(2,[0,0,c,0.;0,1,-s,0.;1,0,s,0.;1,1,c,0.])),
                    //draw the latex gate into tex file
                Draw    = "\\gate{" + nam + "}"
                )
        (gate theta).Run qs

    /// <summary>
    /// Prepares the initial quantum state as required by Schuld's amplitude-based k-nearest neighbour algorithm. 
    /// </summary>
    /// <param name="qs">The qubit list with four qubits that is being manipulated.</param>
    let statepreparation (qs:Qubits) =

        //Preparing the up and down state as training vectors
        //The |+> state is used as new input vector

        //extract the individual qubits
        let q0, q1, q2, q3 = qs.Head, qs.[1], qs.[2], qs.[3]

        //prepare superposition to separate training and input vectors
        H   [q0]
        //put m register into superposition
        H   [q3]

        ///// PREPARING THE NEW INPUT VECTOR
        //controlled Sdagger (q0 control, q1 target)
        Cgate (Adj S) [q0;q1]
        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        //controlled Tdagger (q0 control, q1 target)
        Cgate (Adj T) [q0;q1]
        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        ////////

        //flip the class label with CNOT (q3 control, q2 target)
        CNOT [q3;q2]
        //flip the first qubit >> move input vector to the front
        X   [q0]
        //apply Toffoli (q0 & q3 controls, q1 target) >> to create the second training vector
        CCNOT   [q0;q3;q1]

    /// <summary>
    /// Collects the statistics for a qubit list with four qubits. 
    /// </summary>
    /// <param name="qs">The qubit list of which the statistics shall be calculated from.</param>
    /// <param name="stats">A float array with 16 items storing the stats for the states |0000>, |0001>, etc.</param>
    /// <param name="stats01">A float array with 8 items collecting the stats for the individual qubits (ancilla, data, class and m)</param>
    let collectstats (qs:Qubits) (stats:_[]) (stats01:_[]) = 
            //info on how to pass arrays into functions: http://stackoverflow.com/questions/16968060/f-why-cant-i-access-the-item-member

            //measure all the qubits in the z-basis
            M >< qs

            //retrieve the bit values of the qubits in qubit list qs and convert to integer (v)
            let v,w,x,y   = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v

            //dot between stats and brackets is needed when retrieving objects from a list or array
            stats01.[v] <- stats01.[v] + 1.0 //reaches items 0 or 1 >> ancilla qubit stats
            stats01.[2+w] <- stats01.[2+w] + 1.0 //reaches item 2 or 3 >> data qubit stats
            stats01.[4+x] <- stats01.[4+x] + 1.0 //reaches item 4 or 5 >> class qubit stats
            stats01.[6+y] <- stats01.[6+y] + 1.0 //reaches item 6 or 7 >> m qubit stats
            
            //match qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v with
            match v,w,x,y with
                | 0,0,0,0 -> stats.[0] <- stats.[0] + 1.0
                | 0,0,0,1 -> stats.[1] <- stats.[1] + 1.0
                | 0,0,1,0 -> stats.[2] <- stats.[2] + 1.0
                | 0,1,0,0 -> stats.[3] <- stats.[3] + 1.0
                | 1,0,0,0 -> stats.[4] <- stats.[4] + 1.0
                | 0,0,1,1 -> stats.[5] <- stats.[5] + 1.0
                | 0,1,1,0 -> stats.[6] <- stats.[6] + 1.0
                | 1,1,0,0 -> stats.[7] <- stats.[7] + 1.0
                | 1,0,0,1 -> stats.[8] <- stats.[8] + 1.0
                | 0,1,0,1 -> stats.[9] <- stats.[9] + 1.0
                | 1,0,1,0 -> stats.[10] <- stats.[10] + 1.0
                | 0,1,1,1 -> stats.[11] <- stats.[11] + 1.0
                | 1,1,1,0 -> stats.[12] <- stats.[12] + 1.0
                | 1,1,0,1 -> stats.[13] <- stats.[13] + 1.0
                | 1,0,1,1 -> stats.[14] <- stats.[14] + 1.0
                | 1,1,1,1 -> stats.[15] <- stats.[15] + 1.0
                | _,_,_,_ -> show "error" //to handle all other cases (which won't occur any way)
    
    /// <summary>
    /// Prints the individual statistics of 4 qubits and their combinations (|0000>, |0001>, etc.).
    /// </summary>
    /// <param name="stats">A float array with 16 items storing the stats for the states |0000>, |0001>, etc.</param>
    /// <param name="stats01">A float array with 8 items collecting the stats for the individual qubits (ancilla, data, class and m)</param>
    let printstats (stats:_[]) (stats01:_[]) =

        //printing etiquette:
        //printfn "A string: %s. An int: %i. A float: %f. A bool: %b" "hello" 42 3.14 true

        show "Measured ancilla qubit: |0>: %f |1>: %f" stats01.[0] stats01.[1]
        //show "Old Measured ancilla qubit: 0-%d 1-%d" stats0.[0] stats0.[1]
        show "Measured data register: |0> %f |1> %f" stats01.[2] stats01.[3]
        //show "Old Measured data qubit: 0-%d 1-%d" stats1.[0] stats1.[1]
        show "Measured class register: |0> %f |1> %f" stats01.[4] stats01.[5]
        //show "Old Measured class qubit: 0-%d 1-%d" stats2.[0] stats2.[1]
        show "Measured m register: |0> %f |1> %f" stats01.[6] stats01.[7]
        //show "Old Measured m qubit: 0-%d 1-%d" stats3.[0] stats3.[1]
        show "Measured |0000>: %f" stats.[0]
        show "Measured |0001>: %f" stats.[1]
        show "Measured |0010>: %f" stats.[2]
        show "Measured |0100>: %f" stats.[3]
        show "Measured |1000>: %f" stats.[4]
        show "Measured |0011>: %f" stats.[5]
        show "Measured |0110>: %f" stats.[6]
        show "Measured |1100>: %f" stats.[7]
        show "Measured |1001>: %f" stats.[8]
        show "Measured |0101>: %f" stats.[9]
        show "Measured |1010>: %f" stats.[10]
        show "Measured |0111>: %f" stats.[11]
        show "Measured |1110>: %f" stats.[12]
        show "Measured |1101>: %f" stats.[13]
        show "Measured |1011>: %f" stats.[14]
        show "Measured |1111>: %f" stats.[15]

    [<LQD>] //means it can be called from the command line
    let __UserSample(n:int) = //n defines the number of runs
    //let __UserSample() =

        //initialize statistic arrays
        //float array with 16 items and initialize with 0.0 >> will hold the stats for the combination states like |0000>, |0001>, etc.
        let stats  = Array.create 16 0.0
        //float array with 8 items to store the stats of the individual qubits
        let stats01  = Array.create 8 0.0
        //let stats0  = Array.create 2 0
        //let stats1  = Array.create 2 0
        //let stats2  = Array.create 2 0
        //let stats3  = Array.create 2 0

        //create state vector containing a single qubit
        let k  = Ket(4)
        let qs = k.Qubits

        //create circuit
        let circ = Circuit.Compile statepreparation qs

        //output the circuit into the log file
        circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        circ.RenderHT("StatePreparation")

        //convolutes the quantum gates >> impossible on an actual quantum computer
        //but leads to a speed up in the classical simulation
        //e.g. collapsing 9 CNOT gates into one matrix >> speed up!
        let circ    = circ.GrowGates(k)

        //output the circuit into the log file
        circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        circ.RenderHT("StatePreparationOptim")

        for i in 0..(n-1) do

            //reset the state vector since the measurement will collapse the state vector
            let qs = k.Reset()

            //instead of 'statepreparation qs' I run the circuit since it was optimized by the GrowGates algorithm
            circ.Run qs

            //interfere the training vectors with the new input vector
            H   qs

            ////CONDITIONAL MEASUREMENT SHOULD COME HERE

            //collect the statistics
            collectstats qs stats stats01

        //divide the qubit counts by the number of runs
        for s in 0..15 do
            stats.[s] <- stats.[s]/float(n)
            if s < 8 then
                stats01.[s] <- stats01.[s]/float(n)
        
        printstats stats stats01

        /////////////// OLD CODE SNIPPETS ///////////////////////

        //PRINTING THE QUBITS
        //for q in qs do
        //show "q0 = %s" (q0.ToString())

        //TUTORIAL SNIPPETS:
        //rotate qubit by 90 degrees in x direction (Math.PI/2. different direction than H rotation)
        //rotX (Math.PI/4.)   qs //do this to first qubit

        //Entanglement
        //for q in qs.Tail do CNOT [qs.Head;q]//select the other qubits but the first one
        //apply a CNOT between the head qubit and each other qubit >> results in entanglement

        //M does measurement of one qubit (it does the first one in the list)
        //M >< qs //bow tie operator applies measurement to all qubits

        //show "q = %s" (qs.[0].ToString())
        //prepare the initial state by running the circuit

        //statepreparation    qs
        //show "qaH = %s" (qs.ToString())
    
        //show "test1:"
        //output the circuit into the log file
        //circ.Dump()
        //Draw it into HTML(H) and Tex (T)
        //circ.RenderHT("Test1")

        //show "test2:"
        //output the circuit into the log file
        //circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        //circ.RenderHT("Test2")

        //OLD way of counting the statistic of the individual qubits
        //let v,w,x,y   = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v
        //stats0.[v] <- stats0.[v] + 1
        //stats1.[w] <- stats1.[w] + 1
        //stats2.[x] <- stats2.[x] + 1
        //stats3.[y] <- stats3.[y] + 1

        //Test for entanglement
        //for q in qs.Tail do
            //if q.Bit <> qs.Head.Bit then
                //failwith "BAD!!!!!"

module Main =
    open App

    /// <summary>
    /// The main entry point for Liquid.
    /// </summary>
    [<EntryPoint>]
    let Main _ =
        RunLiquid ()
