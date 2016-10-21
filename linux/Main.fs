namespace Microsoft.Research.Liquid

module BinaryAmplitudeKNN =
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
    /// Prepare the 3/4 state for the x-z plane classifier
    /// </summary>
    /// <param name="qs">The qubits that will be prepared.</param>
    let state1 (q0:Qubit) (q1:Qubit) (q2:Qubit) (q3:Qubit) =

        //// #1 \\\\\
        ////////// START: Binary classifier in z-x-plane \\\\\\\\\\\\\\\\\\\\\\
        /// input ector (3/4) >> alpha = 0.85355 - 0.35355i; beta = 0.35355 - 0.14645i
        /// training vector #1 >> alpha = 1; beta = 0
        /// training vector #2 >> alpha = 0; beta = 1

        //prepared state > probabilities checked twice!
        //interfered state >> probabilities checked twice!
        //python bloch sphere mapping >> available!

        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        //controlled Tdagger (q0 control, q1 target)
        Cgate (Adj T) [q0;q1]
        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        //controlled Sdagger (q0 control, q1 target)
        Cgate (Adj S) [q0;q1]

        //OPTIONAL:
        //flip it such that it should be classified as |1>
        //CNOT [q0;q1]

        //flip the class label with CNOT (q3 control, q2 target)
        CNOT [q3;q2]
        //flip the first qubit >> move input vector to the front
        X   [q0]
        //apply Toffoli (q0 & q3 controls, q1 target) >> to create the second training vector
        CCNOT   [q0;q3;q1]
        //////// END \\\\\\\\\

     /// <summary>
    /// Prepare the 7/8 state for the x-z plane classifier
    /// </summary>
    /// <param name="qs">The qubits that will be prepared.</param>
    let state2 (q0:Qubit) (q1:Qubit) (q2:Qubit) (q3:Qubit) =

        //// #2 \\\\\
        ////////// START: Binary classifier in z-x-plane \\\\\\\\\\\\\\\\\\\\\\
        /// input vector (7/8) >> alpha = 0.96194 - 0.19134i; beta = 0.19134 - 0.03806i
        /// training vector #1 >> alpha = 1; beta = 0
        /// training vector #2 >> alpha = 0; beta = 1

        //prepared state > probabilities checked!
        //interfered state >> probabilities checked!
        //python bloch sphere mapping >> available!

        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        //controlled Tdagger (q0 control, q1 target)
        Cgate (Adj T) [q0;q1]
        //controlled pi/8 rotation (q0 control, q1 target)
        Cgate (R 4) [q0;q1]
        //controlled H (q0 control, q1 target)
        Cgate H [q0;q1]
        //controlled Sdagger (q0 control, q1 target)
        Cgate (Adj S) [q0;q1]

        //if needed:
        //more controlled rotations
        //Cgate (Adj (R 5)) [q0;q1]
        //Cgate (Adj (R 6)) [q0;q1]

        //OPTIONAL:
        //flip it such that it should be classified as |1>
        //CNOT [q0;q1]

        //flip the class label with CNOT (q3 control, q2 target)
        CNOT [q3;q2]
        //flip the first qubit >> move input vector to the front
        X   [q0]
        //apply Toffoli (q0 & q3 controls, q1 target) >> to create the second training vector
        CCNOT   [q0;q3;q1]
        //////// END \\\\\\\\\

    /// <summary>
    /// Prepare the 3/4 state for the x-y plane classifier
    /// </summary>
    /// <param name="qs">The qubits that will be prepared.</param>
    let state3 (q0:Qubit) (q1:Qubit) (q2:Qubit) (q3:Qubit) =

        //// #3 \\\\\
        ////////// START: Binary classifier in x-y-plane \\\\\\\\\\\\\\\\\\\\\\
        /// input vector >> alpha = 0.70711; beta = 0.50000 + 0.50000i
        /// training vector #1 >> alpha = 0.70711; beta = 0.70711i
        /// training vector #2 >> alpha = 0.70711; beta = -0.70711i

        //prepared state > probabilities checked!
        //interfered state >> probabilities checked!
        //python bloch sphere mapping >> available!

        //put the second qubit into |+> state
        H [q1]
        //Prepare input vector: controlled T (q0 control, q1 target)
        Cgate T [q0;q1]

        //OPTIONAL:
        //flip it such that it should be classified as |1>
        //Cgate Z [q0;q1]

        //flip the first qubit >> move input vector to the front
        X   [q0]
        //controlled controlled S gate (q0 & q3 control, q2 target)
        //creates the first training vector
        Cgate S [q0;q1]
        //controlled controlled Z to flip the phase in which is to become the second training vector
        //separates the first from the second training vector
        CCgate  Z   [q0;q3;q1]
        //flip the class label with CNOT (q3 control, q2 target)
        CNOT [q3;q2]
        //////// END \\\\\\\\\

    /// <summary>
    /// Prepares the initial quantum state as required by Schuld's amplitude-based k-nearest neighbour algorithm.
    /// </summary>
    /// <param name="qs">The qubit list with four qubits that is being manipulated.</param>
    let statepreparation (qs:Qubits) =
        //OPTIONS
        //STATE 1 -> X-Z plane binary classifier with 3/4 state
        //STATE 2 -> X-Z plane binary classifier with 7/8 state
        //STATE 3 -> X-Y plane binary classifier with 3/4 state
        let state = 3

        //extract the individual qubits
        let q0, q1, q2, q3 = qs.Head, qs.[1], qs.[2], qs.[3]

        //prepare superposition to separate training and input vectors
        H   [q0]
        //put m register into superposition
        H   [q3]

        ///// PREPARING the qubits in the selected state (state preparation functions are defined above)
        match state with
            | 1 -> state1 q0 q1 q2 q3
            | 2 -> state2 q0 q1 q2 q3
            | 3 -> state3 q0 q1 q2 q3
            | _ -> show "option undefined"

    /// <summary>
    /// Collects the statistics for a qubit list with four qubits.
    /// </summary>
    /// <param name="qs">The qubit list of which the statistics shall be calculated from.</param>
    /// <param name="stats">A float array with 16 items storing the stats for the states |0000>, |0001>, etc.</param>
    /// <param name="stats01">A float array with 8 items collecting the stats for the individual qubits (ancilla, data, class and m)</param>
    let collectstats (qs:Qubits) (stats:_[]) (stats01:_[]) (conditional:bool) =
            //info on how to pass arrays into functions: http://stackoverflow.com/questions/16968060/f-why-cant-i-access-the-item-member

            //measure all the qubits in the z-basis
            if conditional = true then
                M   [qs.[1]]
                M   [qs.[2]]
                M   [qs.[3]]
            else
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
        if stats01.[4] > stats01.[5] then
            show "Input classified as: |0>"
        else
            show "Input classified as: |1>"

    /// <summary>
    /// Outsourcing the second part of the amplitude-KNN algorithm to create a TEX and HTML file.
    /// </summary>
    /// <param name="qs">Qubit list</param>
    let secondpartofALG (qs:Qubits) =

            H   qs
            X   qs
            BC M [qs.[0];qs.[2]]


    [<LQD>] //means it can be called from the command line
    let __BinaryAmplitudeKNN(runs:int) = //parameter defines the number of runs
        show "The quantum kNN based on amplitude encoding"
        show "_______________________________________________________"
        //OPTIONS:
        //if only statepreparation is wanted >> false, false
        //if statepreparation and interference is wanted >> false, true
        //if full classification is wanted >> true, trues
        let CM = true
        let interfere = true

        //initialize statistic arrays
        //float array with 16 items and initialize with 0.0 >> will hold the stats for the combination states like |0000>, |0001>, etc.
        let stats  = Array.create 16 0.0
        //float array with 8 items to store the stats of the individual qubits
        let stats01  = Array.create 8 0.0
        //let stats0  = Array.create 2 0
        //let stats1  = Array.create 2 0
        //let stats2  = Array.create 2 0
        //let stats3  = Array.create 2 0

        let mutable conditionalcounter = 0

        //create state vector containing a single qubit
        let k  = Ket(4)
        let qs = k.Qubits

        //create circuit
        let circ = Circuit.Compile statepreparation qs
        let circ2 = Circuit.Compile secondpartofALG qs

        let totalcirc = Seq [circ;circ2]
        totalcirc.RenderHT("TotalCircuit")

        totalcirc.Dump()

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

        for i in 0..(runs-1) do

            //reset the state vector since the measurement will collapse the state vector
            let qs = k.Reset()

            //instead of 'statepreparation qs' I run the circuit since it was optimized by the GrowGates algorithm
            circ.Run qs

            //secondpartofALG qs interfere CM stats stats01 conditionalcounter n

            //interfere the training vectors with the new input vector
            if interfere = true then
                H   qs

            if CM = true then
                //X   [qs.[0]]
                //BC M [qs.[0];qs.[2]]
                //need to prevent the unknown bit case!
                //if qs.[0].Bit.v = 1 then
                    //collectstats qs stats stats01 true
                   // conditionalcounter <- conditionalcounter + 1


            ////CONDITIONAL MEASUREMENT on q0
                M   [qs.[0]]
                if qs.[0].Bit.v = 0 then
                    collectstats qs stats stats01 true
                    conditionalcounter <- conditionalcounter + 1
            else
                //collect the statistics
                collectstats qs stats stats01 false
                conditionalcounter <- runs

        //divide the qubit counts by the number of runs
        for s in 0..15 do
            stats.[s] <- stats.[s]/float(conditionalcounter)
            if s < 8 then
                stats01.[s] <- stats01.[s]/float(conditionalcounter)

        printstats stats stats01
        show "Successful CMs: %i" conditionalcounter
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

module qubitKNN =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates----\\

    let U1 (qs:Qubits) =
        let gate (qs:Qubits) =
            Gate.Build("U1", fun () ->
                new Gate(
                    Qubits = qs.Length,
                    Name = "U1",
                    Help = "First part of the Hamiltonian unitary",
                    //Draw = .....,
                    Op = WrapOp (fun (qs:Qubits) ->
                            X   qs;
                            Adj (R 5) qs;
                            X   qs;
                            )
            ))
        (gate qs).Run qs

    let U2 (qs:Qubits) =
        let gate =
            Gate.Build("U2", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "U2",
                    Help = "Second part of the Hamiltonian unitary",
                    Mat  = (CSMat(2,[(0,0,0.92388,0.38268); (1,1,1.,0.)]))
                    //Draw = .....,
            ))
        gate.Run qs


    let HDUnitary (qs:Qubits) =
        let gate (qs:Qubits) =
            Gate.Build("HDUnitary", fun () ->
                let nam     = "HDUnitary"
                new Gate(
                    Qubits = qs.Length,
                    Name = "HDUnitary",
                    Help = "Sums up the Hamming distances",
                    //Draw = .....,
                    Op = WrapOp (fun (qs:Qubits) ->
                            U1          qs;
                            Cgate U2    qs;
                            )
            ))
        (gate qs).Run qs

    //-----END: define new gates----\\

    //-----START: define new functions----\\

    let statepreparation (qs:Qubits) =

        //INITIALIZE FIRST REGISTER >> INPUT VECTOR in decimals = [0.6;0.4]
        //apply NOT gates where the classical bit string has ones
        X   [qs.[1]]
        X   [qs.[2]]
        X   [qs.[5]]

        //put class qubit into superposition
        H   [qs.[16]]

        //now flip the respective qubits using CNOT gates
        //first training vector >> class qubit used as control
        CNOT    [qs.[16];qs.[8]]
        CNOT    [qs.[16];qs.[10]]

        //move the first training vector to the front by flipping the class label
        X       [qs.[16]]

        //second training vector >> class qubit used as control
        CNOT    [qs.[16];qs.[12]]
        CNOT    [qs.[16];qs.[14]]

    let SchuldQMLAlg (qs:Qubits) =

            //Put ancilla register into superposition
            H   [qs.[17]]

            //----------------------------------------
            // Calculate the Hamming distance quantum mechanically!

            for i in 0..7 do
                //CNOT calculates the Hamming distance
                CNOT    [qs.[i];qs.[i+8]]
                //reverse the Hamming distance
                X   [qs.[i+8]]

            //------- Applying the Hamiltonian operator to sum the Hamming distances -----\\

            for j in 0..7 do
              //apply the unitary operator >> use the ancilla qubit as control for the CU^(-2) operation (see Trugenberger et al., 2001)
              HDUnitary   [qs.[17];qs.[8+j]]

            //Hadamard on ancilla writes the total Hamming distance into the amplitudes
            H   [qs.[17]]

            //measuring the ancilla qubit
            M   [qs.[17]]


    //-----END: define new functions----\\

    [<LQD>] //means it can be called from the command line
    let __qubitKNN(runs:int) =
        show "The quantum kNN based on qubit encoding"
        show "_______________________________________________________"

        //initialize stats arrays
        let stats = Array.create 2 0
        let cstats = Array.create 2 0

        //First register: inputvector is a classical 8 bit (1 byte) and a quantum mechanical 8 qubit string
        //Second register: 8 qubits for the training vectors and 1 qubit for the class label
        //Third register: 1 qubit as an ancilla
        //Total: 18 qubits needed
        let k = Ket(18)
        let initialstate = k.Qubits

        //Circuit business
        let preparecirc = Circuit.Compile statepreparation initialstate
        preparecirc.Dump() //output into log file
        //Draw it into HTML (H) and Tex (T)
        preparecirc.Fold().RenderHT("qubitKNNStatePreparation")
        //convolutes the quantum gates >> impossible on an actual quantum computer


        let algorithmcirc = Circuit.Compile SchuldQMLAlg initialstate
        algorithmcirc.Dump() //output into log file
        //Draw it into HTML (H) and Tex (T)
        algorithmcirc.Fold().RenderHT("qubitKNNAlgorithm")
        //convolutes the quantum gates >> impossible on an actual quantum computer

        let totalcirc = Seq [preparecirc;algorithmcirc]
        totalcirc.Dump()
        totalcirc.Fold().RenderHT("TotalCircuit")

        let preparecirc    = preparecirc.GrowGates(k)
        preparecirc.Run initialstate
        let algorithmcirc = algorithmcirc.GrowGates(k)

        /////// START OF THE ACTUAL QML ALGORITHM \\\\\\\\

        for l in 0..(runs-1) do

            show "%i" l
            let initialstate = k.Reset()

            algorithmcirc.Run initialstate

            (*
            //Put ancilla register into superposition
            H   [initialstate.[17]]

            //----------------------------------------
            // Calculate the Hamming distance quantum mechanically!

            for i in 0..7 do
                //CNOT calculates the Hamming distance
                CNOT    [initialstate.[i];initialstate.[i+8]]
                //reverse the Hamming distance
                X   [initialstate.[i+8]]

            //------- Applying the Hamiltonian operator to sum the Hamming distances -----\\

            for i in 0..7 do
              //apply the unitary operator >> use the ancilla qubit as control for the CU^(-2) operation (see Trugenberger et al., 2001)
              HDUnitary   [initialstate.[17];initialstate.[8+i]]

            //Hadamard on ancilla writes the total Hamming distance into the amplitudes
            H   [initialstate.[17]]

            //measuring the ancilla qubit
            M   [initialstate.[17]]*)


            //retrieve the ancilla stats
            let w = initialstate.[17].Bit.v
            stats.[w] <- stats.[w] + 1

            //Conditional measurement (CM)
            if w = 0 then
              M [initialstate.[16]]
              //if CM was successful measure the class qubits
              let c = initialstate.[16].Bit.v
              cstats.[c] <- cstats.[c] + 1

        //---OUTPUT---\\
        show "Ancilla measured as 0: %d" stats.[0]
        show "Ancilla measured as 1: %d" stats.[1]
        show "Class measured as 0: %d" cstats.[0]
        show "Class measured as 1: %d" cstats.[1]

        if cstats.[0] > cstats.[1] then
          show "Input classified as |0>"
        else
          show "Input classied as |1>"

        //----OTHER STUFF----\\

        (* IMPORTANT! CNOT cannot be applied to qubits from different states!
        let a  = Ket(4)
        let qs1 = a.Qubits
        let b  = Ket(4)
        let qs2 = b.Qubits
        let tester = !!(qs1,qs2)

        CNOT    [tester.[0];tester.[4]]
        *)


module Main =
    open App

    /// <summary>
    /// The main entry point for Liquid.
    /// </summary>
    [<EntryPoint>]
    let Main _ =
        RunLiquid ()
