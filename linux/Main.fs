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
        /// input vector (3/4) >> alpha = 0.85355 - 0.35355i; beta = 0.35355 - 0.14645i
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
        let state = 1

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

        //show "Measured ancilla qubit: |0>: %f |1>: %f" stats01.[0] stats01.[1]
        //show "Old Measured ancilla qubit: 0-%d 1-%d" stats0.[0] stats0.[1]
        //show "Measured data register: |0> %f |1> %f" stats01.[2] stats01.[3]
        //show "Old Measured data qubit: 0-%d 1-%d" stats1.[0] stats1.[1]
        show "Measured class register: |0> %f |1> %f" stats01.[4] stats01.[5]
        //show "Old Measured class qubit: 0-%d 1-%d" stats2.[0] stats2.[1]
        //show "Measured m register: |0> %f |1> %f" stats01.[6] stats01.[7]
        //show "Old Measured m qubit: 0-%d 1-%d" stats3.[0] stats3.[1]
        (*
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
        *)
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

        //create state vector containing four qubits
        let k  = Ket(4)
        let qs = k.Qubits

        //create circuit
        let circ = Circuit.Compile statepreparation qs
        let circ2 = Circuit.Compile secondpartofALG qs

        let totalcirc = Seq [circ;circ2]
        //totalcirc.RenderHT("TotalCircuit")

        //totalcirc.Dump()

        //output the circuit into the log file
        //circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        //circ.RenderHT("StatePreparation")

        //convolutes the quantum gates >> impossible on an actual quantum computer
        //but leads to a speed up in the classical simulation
        //e.g. collapsing 9 CNOT gates into one matrix >> speed up!
        let circ    = circ.GrowGates(k)

        //output the circuit into the log file
        //circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        //circ.RenderHT("StatePreparationOptim")

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

        show "Probability of successful CM: %f" (float(conditionalcounter)/float(runs))
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
                            X   qs; Adj (R 5) qs; X   qs;
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

        //second training vector >> class qubit again used as control
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
        let quantumstate = k.Qubits

        //----Circuit initilizations----\\

        //State preparation circuit
        let preparecirc = Circuit.Compile statepreparation quantumstate
        preparecirc.Dump() //output into log file
        preparecirc.Fold().RenderHT("qubitKNNStatePrparation") //Draw it into HTML (H) and Tex (T)

        //Circuit for the actual algorithm
        let algorithmcirc = Circuit.Compile SchuldQMLAlg quantumstate
        algorithmcirc.Dump()
        algorithmcirc.Fold().RenderHT("qubitKNNAlgorithm")

        //Combine both circuits
        let totalcirc = Seq [preparecirc;algorithmcirc]
        totalcirc.Dump()
        totalcirc.Fold().RenderHT("TotalCircuit")

        //convolutes the quantum gates >> impossible on an actual quantum computer
        let preparecirc    = preparecirc.GrowGates(k)
        //prepare the
        preparecirc.Run quantumstate
        let algorithmcirc = algorithmcirc.GrowGates(k)

        /////// START OF THE ACTUAL QML ALGORITHM \\\\\\\\

        for l in 0..(runs-1) do

            show "%i" l
            let quantumstate = k.Reset()

            algorithmcirc.Run quantumstate

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
            let w = quantumstate.[17].Bit.v
            stats.[w] <- stats.[w] + 1

            //Conditional measurement (CM)
            if w = 0 then
              //if CM was successful measure the class qubit
              M [quantumstate.[16]]
              let c = quantumstate.[16].Bit.v
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

module TrugenbergerStorage =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates ----\\

    //IMPORTANT TAKE HOME MESSAGE FROM THE DOCUMENTATION: The function cannot be
    // used if the gate created could be different on each instantiation. The gate creation function is only
    // called once and from then on is looked up in a dictionary.

    // general Trugenberger S matrix:
    // Mat  = (CSMat(2,[(0,0,sqrt((p-i)/(p+1.-i)),0.); (1,1,sqrt((p-i)/(p+1.-i)),0.);(1,0,-1./sqrt(p+1.-i),0.); (0,1,1./sqrt(p+1.-i),0.)]))

    let TrugenbergerS1 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS1", fun () ->
                new Gate(
                    Name = "TrugenbergerS1",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=2",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-1.)/(p)),0.); (1,1,sqrt((p-1.)/(p)),0.);(1,0,-1./sqrt(p),0.); (0,1,1./sqrt(p),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS2 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS2", fun () ->
                new Gate(
                    Name = "TrugenbergerS2",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=2",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-2.)/(p-1.)),0.); (1,1,sqrt((p-2.)/(p-1.)),0.);(1,0,-1./sqrt(p-1.),0.); (0,1,1./sqrt(p-1.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS3 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS3", fun () ->
                new Gate(
                    Name = "TrugenbergerS3",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=3",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-3.)/(p-2.)),0.); (1,1,sqrt((p-3.)/(p-2.)),0.);(1,0,-1./sqrt(p-2.),0.); (0,1,1./sqrt(p-2.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS4 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS4", fun () ->
                new Gate(
                    Name = "TrugenbergerS4",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=4",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-4.)/(p-3.)),0.); (1,1,sqrt((p-4.)/(p-3.)),0.);(1,0,-1./sqrt(p-3.),0.); (0,1,1./sqrt(p-3.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS5 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS5", fun () ->
                new Gate(
                    Name = "TrugenbergerS5",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=5",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-5.)/(p-4.)),0.); (1,1,sqrt((p-5.)/(p-4.)),0.);(1,0,-1./sqrt(p-4.),0.); (0,1,1./sqrt(p-4.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS6 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS6", fun () ->
                new Gate(
                    Name = "TrugenbergerS6",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=6",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-6.)/(p-5.)),0.); (1,1,sqrt((p-6.)/(p-5.)),0.);(1,0,-1./sqrt(p-5.),0.); (0,1,1./sqrt(p-5.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs


    //-----END: define new gates----\\

    //-----START: define new functions----\\

    let nCNOT (patternlength:int) (mregisterstart:int) (u1position:int) (psi:Qubits) =

            match patternlength with
                | 1 -> CNOT  [psi.[mregisterstart];psi.[u1position]]
                | 2 -> CCNOT  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[u1position]]
                | 3 -> Cgate (CCNOT)  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[u1position]]
                | 4 -> Cgate (Cgate (CCNOT))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[u1position]]
                | 5 -> Cgate (Cgate (Cgate (CCNOT)))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[u1position]]
                | 6 -> Cgate (Cgate (Cgate (Cgate (CCNOT))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[u1position]]
                | 7 -> Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[u1position]]
                | 8 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[u1position]]
                | 9 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[mregisterstart+8];psi.[u1position]]
                | 10 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[mregisterstart+8];psi.[mregisterstart+9];psi.[u1position]]
                | 11 -> show "n-CNOT gate not defined yet"
                | 12 -> show "n-CNOT gate not defined yet"
                | 13 -> show "n-CNOT gate not defined yet"
                | 14 -> show "n-CNOT gate not defined yet"
                | 15 -> show "n-CNOT gate not defined yet"
                | 16 -> show "n-CNOT gate not defined yet"
                | _ -> show "n-CNOT gate not defined yet"


    /// <summary>
    /// Collects the statistics for a qubit list with four qubits.
    /// </summary>
    /// <param name="qs">The qubit list of which the statistics shall be calculated from.</param>
    /// <param name="stats">A float array with 16 items storing the stats for the states |0000>, |0001>, etc.</param>
    /// <param name="stats01">A float array with 8 items collecting the stats for the individual qubits (ancilla, data, class and m)</param>
    let collectstats (numberofqubits:int) (patternstorage:string[]) (stats:_[]) (qs:Qubits) =

            //initialize variables and array
            let mutable arraycounter = 0
            let mutable memoryqubitstring = ""
            let memoryqubitvalue = Array.create (patternstorage.[0].Length) 0

            //measure all the qubits
            M >< qs

            for i in numberofqubits/2+1..numberofqubits-1 do
                //only retrieve the bit values of the memory register since this is what we're most interested in!
                memoryqubitvalue.[arraycounter] <- qs.[i].Bit.v
                //create a string of the measured memory register
                memoryqubitstring <- memoryqubitstring + (string memoryqubitvalue.[arraycounter])
                arraycounter <- arraycounter + 1

            for p in 0..patternstorage.Length-1 do
                //increase the stats by 1 for the matching pattern!
                if memoryqubitstring = patternstorage.[p] then
                    stats.[p] <- stats.[p] + 1



    let StorageAlgorithm (patternstorage:string[]) (patternlength:int) (patternnumber:int) (psi:Qubits)=

            //Define the important variables
            let u1position = patternlength
            let u2position = patternlength + 1
            let mregisterstart = u2position + 1

            if patternnumber = 1 then
                for l in 0..patternlength-1 do
                //if the pattern has a 1 at this point then flip the corresponding qubit
                    if patternstorage.[0].[l] = '1' then
                        X   [psi.[l]]
            else
                // LOADING THE SECOND PATTERN
                for k in 0..patternlength-1 do
                //if the pattern has a 1 at this point then flip the corresponding qubit
                    if patternstorage.[patternnumber-2].[k] <> patternstorage.[patternnumber-1].[k] then
                        //show "%i" k
                        X   [psi.[k]]

            // STEP 1
            for i in 0..(patternlength-1) do
                CCNOT [psi.[i];psi.[u2position];psi.[mregisterstart+i]]

            // STEP 2
            for j=0 to patternlength-1 do
                CNOT [psi.[j];psi.[mregisterstart+j]]
                X    [psi.[mregisterstart+j]]

            // STEP 3
            //applying the n-CNOT gate (all memory qubits as controls and u1 as target qubit)
            nCNOT patternlength mregisterstart u1position psi

              // STEP 4
              //separating out the new pattern >> automatically takes care of normalization
            match patternnumber with
                | 1 -> Cgate (TrugenbergerS1 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 2 -> Cgate (TrugenbergerS2 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 3 -> Cgate (TrugenbergerS3 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 4 -> Cgate (TrugenbergerS4 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 5 -> Cgate (TrugenbergerS5 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 6 -> Cgate (TrugenbergerS6 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | _ -> show "Trugenberger's CS^(p+1-i) gate not defined yet"

            // STEP 5
            //undo STEP 3
            nCNOT patternlength mregisterstart u1position psi

            // STEP 6
            //undo STEP 2
            for j in 0..patternlength-1 do
                X    [psi.[mregisterstart+j]]
                CNOT [psi.[j];psi.[mregisterstart+j]]

                // STEP 7
                //undo STEP 1
            for i in 0..(patternlength-1) do
                CCNOT [psi.[i];psi.[u2position];psi.[mregisterstart+i]]


    (*let TrugenbergerCore (runs:int) (patternlength:int) (patternstorage:string[]) (k:Ket) (psi:Qubits)  =

        let u2position = patternlength + 1

        for i in 0..runs-1 do

            show "ITERATION %i" i
            let psi = k.Reset()

            X   [psi.[u2position]] //flip the second utility qubit

            //maybe put this into the StorageAlgorithm function
            //X   [psi.[u2position]] //flip the second utility qubit



            for p in 1..patterncount do
                StorageAlgorithm patternstorage patternlength p psi


            collectstats requiredqubits patternstorage stats psi

        show "========= STATS ========="
        for m in 0..patternstorage.Length-1 do
            show "Measured |%s>: %i" patternstorage.[m] stats.[m]*)

    //-----END: define new functions----\\


    [<LQD>] //means it can be called from the command line
    let __TrugenbergerStorage(runs:int) =
        show "The memory storage algorithm by Trugenberger et al."
        show "_______________________________________________________"

        // RUNNING THE PROGRAM IN VISUAL STUDIO/MONODEVELOP (DEV MODE) OR EXTERNAL CONSOLE?
        // if DEV MODE -> set the variable to true
        let devmode = true

        //Need to dummy initialize these variables to prevent errors
        //Will be redefined below
        let mutable patternstorage = Array.create 2 "empty"
        let mutable patterncount = 0
        let mutable stats = Array.create 1 0

        if devmode = true then
            stats <- Array.create 10 0
            patterncount <- 2
            patternstorage <- Array.create patterncount "empty"
            //Defining the patterns that are to be stored
            //NEED TO BE SAME LENGTH!
            patternstorage.[0] <- "011"
            patternstorage.[1] <- "101"
            //patternstorage.[2] <- "111"
        else
            // Ask the user for the number of patterns
            Console.Write("Number of patterns to store: ")

            // Read user input
            patterncount <- int (Console.ReadLine())

            patternstorage <- Array.create patterncount "empty"
            stats <- Array.create patterncount 0
            for c in 0..patterncount-1 do
                // Ask the user for pattern
                //NEED TO BE SAME LENGTH!
                Console.Write("Pattern {0}: ", (c+1))

                // Read user input
                patternstorage.[c] <- Console.ReadLine()


        //find the length of the patterns
        let patternlength = patternstorage.[0].Length
        show "Binary pattern length: %i" patternlength

        //memory register (patternlength long); loading register (patternlength long); utility register (2 qubits)
        let requiredqubits = 2*patternlength+2
        show "Number of required qubits: %i" requiredqubits
        let k = Ket(requiredqubits)
        let psi = k.Qubits
        let u2position = patternlength + 1

        for i in 0..runs-1 do

            show "ITERATION %i" i
            let psi = k.Reset()

            X   [psi.[u2position]] //flip the second utility qubit

            //maybe put this into the StorageAlgorithm function
            //X   [psi.[u2position]] //flip the second utility qubit*)

            (* LOADING THE FIRST PATTERN
            for l in 0..patternlength-1 do
            //if the pattern has a 1 at this point then flip the corresponding qubit
                if patternstorage.[0].[l] = '1' then
                    X   [psi.[l]]*)

            for p in 1..patterncount do
                StorageAlgorithm patternstorage patternlength p psi

            (*// LOADING THE SECOND PATTERN
            for k in 0..patternlength-1 do
            //if the pattern has a 1 at this point then flip the corresponding qubit
                if patternstorage.[0].[k] <> patternstorage.[1].[k] then
                    //show "%i" k
                    X   [psi.[k]]*)

            collectstats requiredqubits patternstorage stats psi

        show "========= STATS ========="
        for m in 0..patternstorage.Length-1 do
            show "Measured |%s>: %i" patternstorage.[m] stats.[m]

        (*show "Measured |01100011>: %i" stats.[0]
        show "Measured |01101000>: %i" stats.[1]
        show "Measured |10100011>: %i" stats.[2]
        show "Measured |10100101>: %i" stats.[3]
        show "Measured |10101000>: %i" stats.[4]
        show "Measured |11100111>: %i" stats.[5]
        show "Measured |11100011>: %i" stats.[6]
        show "Measured |11100101>: %i" stats.[7]*)



            //show "qaH = %s" (initial.ToString())
            //X initial
            //X initial.Tail
            //(TrugenbergerCS 2. 2.) initial
            //Cgate (TrugenbergerS 1. 2.) [initial.[0];initial.[1]]
            //H initial.Tail

            //show "qaH = %s" (initial.ToString())

            //M >< initial

            //let v,w = initial.[0].Bit.v, initial.[1].Bit.v

            (*if (v=0) && (v = w) then
                stats.[0] <- stats.[0] + 1
            if (v=0) && (w=1) then
                stats.[1] <- stats.[1] + 1
            if (v=1) && (w=0) then
                stats.[2] <- stats.[2] + 1
            if (v=1) && (v = w) then
                stats.[3] <- stats.[3] + 1


        show "Measured |00>: %i" stats.[0]
        show "Measured |01>: %i" stats.[1]
        show "Measured |10>: %i" stats.[2]
        show "Measured |11>: %i" stats.[3]*)


module TrugenbergerSchuld =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates ----\\

    //IMPORTANT TAKE HOME MESSAGE FROM THE DOCUMENTATION: The function cannot be
    // used if the gate created could be different on each instantiation. The gate creation function is only
    // called once and from then on is looked up in a dictionary.

    // general Trugenberger S matrix:
    // Mat  = (CSMat(2,[(0,0,sqrt((p-i)/(p+1.-i)),0.); (1,1,sqrt((p-i)/(p+1.-i)),0.);(1,0,-1./sqrt(p+1.-i),0.); (0,1,1./sqrt(p+1.-i),0.)]))

    let TrugenbergerS1 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS1", fun () ->
                new Gate(
                    Name = "TrugenbergerS1",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=2",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-1.)/(p)),0.); (1,1,sqrt((p-1.)/(p)),0.);(1,0,-1./sqrt(p),0.); (0,1,1./sqrt(p),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS2 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS2", fun () ->
                new Gate(
                    Name = "TrugenbergerS2",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=2",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-2.)/(p-1.)),0.); (1,1,sqrt((p-2.)/(p-1.)),0.);(1,0,-1./sqrt(p-1.),0.); (0,1,1./sqrt(p-1.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS3 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS3", fun () ->
                new Gate(
                    Name = "TrugenbergerS3",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=3",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-3.)/(p-2.)),0.); (1,1,sqrt((p-3.)/(p-2.)),0.);(1,0,-1./sqrt(p-2.),0.); (0,1,1./sqrt(p-2.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS4 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS4", fun () ->
                new Gate(
                    Name = "TrugenbergerS4",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=4",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-4.)/(p-3.)),0.); (1,1,sqrt((p-4.)/(p-3.)),0.);(1,0,-1./sqrt(p-3.),0.); (0,1,1./sqrt(p-3.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS5 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS5", fun () ->
                new Gate(
                    Name = "TrugenbergerS5",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=5",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-5.)/(p-4.)),0.); (1,1,sqrt((p-5.)/(p-4.)),0.);(1,0,-1./sqrt(p-4.),0.); (0,1,1./sqrt(p-4.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let TrugenbergerS6 (p:float) (qs:Qubits) =
        let gate =
            Gate.Build("TrugenbergerS6", fun () ->
                new Gate(
                    Name = "TrugenbergerS6",
                    Help = "Trugenberger et al.'s S^(p+1-i) matrix with i=6",
                    Mat  = (CSMat(2,[(0,0,sqrt((p-6.)/(p-5.)),0.); (1,1,sqrt((p-6.)/(p-5.)),0.);(1,0,-1./sqrt(p-5.),0.); (0,1,1./sqrt(p-5.),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let U1 (qs:Qubits) =
        let gate (qs:Qubits) =
            Gate.Build("U1", fun () ->
                new Gate(
                    Qubits = qs.Length,
                    Name = "U1",
                    Help = "First part of the Hamiltonian unitary",
                    //Draw = .....,
                    Op = WrapOp (fun (qs:Qubits) ->
                            X   qs; Adj (R 5) qs; X   qs;
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

    let nCNOT (patternlength:int) (mregisterstart:int) (u1position:int) (psi:Qubits) =

            match patternlength with
                | 1 -> CNOT  [psi.[mregisterstart];psi.[u1position]]
                | 2 -> CCNOT  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[u1position]]
                | 3 -> Cgate (CCNOT)  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[u1position]]
                | 4 -> Cgate (Cgate (CCNOT))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[u1position]]
                | 5 -> Cgate (Cgate (Cgate (CCNOT)))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[u1position]]
                | 6 -> Cgate (Cgate (Cgate (Cgate (CCNOT))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[u1position]]
                | 7 -> Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[u1position]]
                | 8 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[u1position]]
                | 9 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[mregisterstart+8];psi.[u1position]]
                | 10 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))))  [psi.[mregisterstart];psi.[mregisterstart+1];psi.[mregisterstart+2];psi.[mregisterstart+3];psi.[mregisterstart+4];psi.[mregisterstart+5];psi.[mregisterstart+6];psi.[mregisterstart+7];psi.[mregisterstart+8];psi.[mregisterstart+9];psi.[u1position]]
                | 11 -> show "n-CNOT gate not defined yet"
                | 12 -> show "n-CNOT gate not defined yet"
                | 13 -> show "n-CNOT gate not defined yet"
                | 14 -> show "n-CNOT gate not defined yet"
                | 15 -> show "n-CNOT gate not defined yet"
                | 16 -> show "n-CNOT gate not defined yet"
                | _ -> show "n-CNOT gate not defined yet"

    let nCNOTforClasses (numberofcontrols:int) (memoryregisterstart:int) (controls:int[]) (classqubitposition:int) (psi:Qubits) =

            match numberofcontrols with
                | 1 -> CNOT  [psi.[memoryregisterstart+controls.[0]];psi.[classqubitposition]]
                | 2 -> CCNOT  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[classqubitposition]]
                | 3 -> Cgate (CCNOT)  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[classqubitposition]]
                | 4 -> Cgate (Cgate (CCNOT))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[classqubitposition]]
                | 5 -> Cgate (Cgate (Cgate (CCNOT)))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[classqubitposition]]
                | 6 -> Cgate (Cgate (Cgate (Cgate (CCNOT))))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[memoryregisterstart+controls.[5]];psi.[classqubitposition]]
                | 7 -> Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[memoryregisterstart+controls.[5]];psi.[memoryregisterstart+controls.[6]];psi.[classqubitposition]]
                | 8 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[memoryregisterstart+controls.[5]];psi.[memoryregisterstart+controls.[6]];psi.[memoryregisterstart+controls.[7]];psi.[classqubitposition]]
                | 9 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT)))))))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[memoryregisterstart+controls.[5]];psi.[memoryregisterstart+controls.[6]];psi.[memoryregisterstart+controls.[7]];psi.[memoryregisterstart+controls.[8]];psi.[classqubitposition]]
                | 10 -> Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (Cgate (CCNOT))))))))  [psi.[memoryregisterstart+controls.[0]];psi.[memoryregisterstart+controls.[1]];psi.[memoryregisterstart+controls.[2]];psi.[memoryregisterstart+controls.[3]];psi.[memoryregisterstart+controls.[4]];psi.[memoryregisterstart+controls.[5]];psi.[memoryregisterstart+controls.[6]];psi.[memoryregisterstart+controls.[7]];psi.[memoryregisterstart+controls.[8]];psi.[memoryregisterstart+controls.[9]];psi.[classqubitposition]]
                | 11 -> show "n-CNOT gate not defined yet"
                | 12 -> show "n-CNOT gate not defined yet"
                | 13 -> show "n-CNOT gate not defined yet"
                | 14 -> show "n-CNOT gate not defined yet"
                | 15 -> show "n-CNOT gate not defined yet"
                | 16 -> show "n-CNOT gate not defined yet"
                | _ -> show "n-CNOT gate not defined yet"


    /// <summary>
    /// Collects the statistics for a qubit list with four qubits.
    /// </summary>
    /// <param name="qs">The qubit list of which the statistics shall be calculated from.</param>
    /// <param name="stats">A float array with 16 items storing the stats for the states |0000>, |0001>, etc.</param>
    /// <param name="stats01">A float array with 8 items collecting the stats for the individual qubits (ancilla, data, class and m)</param>
    let collectstats (numberofqubits:int) (patternstorage:string[]) (stats:_[]) (qs:Qubits) =

            //initialize variables and array
            let mutable arraycounter = 0
            let mutable memoryqubitstring = ""
            let memoryqubitvalue = Array.create (patternstorage.[0].Length) 0

            //measure all the qubits
            M >< qs

            for i in numberofqubits/2+1..numberofqubits-1 do
                //only retrieve the bit values of the memory register since this is what we're most interested in!
                memoryqubitvalue.[arraycounter] <- qs.[i].Bit.v
                //create a string of the measured memory register
                memoryqubitstring <- memoryqubitstring + (string memoryqubitvalue.[arraycounter])
                arraycounter <- arraycounter + 1

            for p in 0..patternstorage.Length-1 do
                //increase the stats by 1 for the matching pattern!
                if memoryqubitstring = patternstorage.[p] then
                    stats.[p] <- stats.[p] + 1

    let collectteststats (stats:_[]) (qs:Qubits) =

        M >< qs
        let a,b,c,d,e,f,g,h,i,j = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v, qs.[4].Bit.v, qs.[5].Bit.v, qs.[6].Bit.v, qs.[7].Bit.v,qs.[8].Bit.v, qs.[9].Bit.v
        show "qubits: %i %i %i %i %i %i %i %i %i %i" a b c d e f g h  i j

        (*match a,b,c,d,e,f,g,h with
                | 0,1,0,0,0,0,0,1 -> stats.[0] <- stats.[0] + 1
                | 0,1,0,0,0,0,1,1 -> stats.[1] <- stats.[1] + 1
                | 0,1,0,1,0,1,1,1 -> stats.[2] <- stats.[2] + 1
                | 0,1,0,1,0,1,0,0 -> stats.[3] <- stats.[3] + 1*)
                (*| 1,0,0,1,0,0,1,1,1,1 -> stats.[0] <- stats.[0] + 1
                | 1,0,0,1,0,0,0,0,0,0 -> stats.[1] <- stats.[1] + 1
                | 1,0,0,1,1,0,1,0,1,1 -> stats.[2] <- stats.[2] + 1
                | 1,0,0,1,1,0,1,1,0,1 -> stats.[3] <- stats.[3] + 1*)
                (*| 0,0,1,0 -> stats.[2] <- stats.[2] + 1.0
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
                | 1,1,1,1 -> stats.[15] <- stats.[15] + 1.0*)
                //| _,_,_,_,_,_,_,_ -> show "error" //to handle all other cases (which won't occur any way)

    let StorageAlgorithm (patternstorage:string[]) (patternlength:int) (patternnumber:int) (psi:Qubits)=

            //Define the important variables
            let u1position = patternlength
            let u2position = patternlength + 1
            let mregisterstart = u2position + 1

            if patternnumber = 1 then
                for l in 0..patternlength-1 do
                //if the pattern has a 1 at this point then flip the corresponding qubit
                    if patternstorage.[0].[l] = '1' then
                        X   [psi.[l]]
            else
                // LOADING THE SECOND PATTERN
                for k in 0..patternlength-1 do
                //if the pattern has a 1 at this point then flip the corresponding qubit
                    if patternstorage.[patternnumber-2].[k] <> patternstorage.[patternnumber-1].[k] then
                        //show "%i" k
                        X   [psi.[k]]

            // STEP 1
            for i in 0..(patternlength-1) do
                CCNOT [psi.[i];psi.[u2position];psi.[mregisterstart+i]]

            // STEP 2
            for j=0 to patternlength-1 do
                CNOT [psi.[j];psi.[mregisterstart+j]]
                X    [psi.[mregisterstart+j]]

            // STEP 3
            //applying the n-CNOT gate (all memory qubits as controls and u1 as target qubit)
            nCNOT patternlength mregisterstart u1position psi

              // STEP 4
              //separating out the new pattern >> automatically takes care of normalization
            match patternnumber with
                | 1 -> Cgate (TrugenbergerS1 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 2 -> Cgate (TrugenbergerS2 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 3 -> Cgate (TrugenbergerS3 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 4 -> Cgate (TrugenbergerS4 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 5 -> Cgate (TrugenbergerS5 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | 6 -> Cgate (TrugenbergerS6 (float patternstorage.Length))  [psi.[u1position];psi.[u2position]]
                | _ -> show "Trugenberger's CS^(p+1-i) gate not defined yet"

            // STEP 5
            //undo STEP 3
            nCNOT patternlength mregisterstart u1position psi

            // STEP 6
            //undo STEP 2
            for j in 0..patternlength-1 do
                X    [psi.[mregisterstart+j]]
                CNOT [psi.[j];psi.[mregisterstart+j]]

                // STEP 7
                //undo STEP 1
            for i in 0..(patternlength-1) do
                CCNOT [psi.[i];psi.[u2position];psi.[mregisterstart+i]]

    let SchuldQMLAlg (inputregisterend:int) (memoryregisterstart:int) (classqubitposition:int) (ancillaqubitposition:int) (qs:Qubits) =

            //Put ancilla register into superposition
            H   [qs.[ancillaqubitposition]]

            //----------------------------------------
            // Calculate the Hamming distance quantum mechanically!

            for i in 0..inputregisterend do
                //CNOT calculates the Hamming distance
                CNOT    [qs.[i];qs.[memoryregisterstart+i]]
                //reverse the Hamming distance
                X   [qs.[memoryregisterstart+i]]

            //------- Applying the Hamiltonian operator to sum the Hamming distances -----\\

            for j in 0..inputregisterend do
              //apply the unitary operator >> use the ancilla qubit as control for the CU^(-2) operation (see Trugenberger et al., 2001)
              HDUnitary   [qs.[ancillaqubitposition];qs.[memoryregisterstart+j]]

            //Hadamard on ancilla writes the total Hamming distance into the amplitudes
            H   [qs.[ancillaqubitposition]]

            //measuring the ancilla qubit
            M   [qs.[ancillaqubitposition]]

    //-----END: define new functions----\\


    [<LQD>] //means it can be called from the command line
    let __TrugenbergerSchuld(runs:int) =
        show "Trugenberger's (2001) storage algorithm combined with Schuld et al.(2014) qubit KNN"
        show "____________________________________________________________________________________"

        // ---- GENERAL INFO ---- \\

        //THIS ALGORITHM IS BASED ON THE PAPER "QUANTUM COMPUTING FOR PATTERN CLASSIFICATION" BY SCHULD ET AL. (2014)
        //PAPER CAN BE FOUND AT: http://arxiv.org/abs/1412.3646
        //THE CODE WAS WRITTEN BY MARK FINGERHUTH FOR HIS BACHELOR THESIS AT THE MAASTRICHT SCIENCE PROGRAMME, MAASTRICHT UNIVERSITY

        // ---- DEVELOPER MODE ---- \\
        // RUNNING THE PROGRAM IN VISUAL STUDIO/MONODEVELOP (DEV MODE) OR EXTERNAL CONSOLE?
        // if DEV MODE -> set the variable to true
        let devmode = false
        let test = false

        // ---- INITIALIZING VARIABLES AND ARRAYS ---- \\
        let mutable trainingpatternstorage = Array.create 1 "empty"
        let mutable classlabelstorage = Array.create 1 "empty"
        let inputpatternstorage = Array.create 1 "empty"
        let mutable stats = Array.create 1 0
        let cstats = Array.create 2 0
        let mutable hd = Array.create 2 0.
        let mutable expect = Array.create 1 "empty"
        let mutable trainingpatterncount = 0
        let mutable conditionalcounter = 0

        // ---- DEFINING THE TRAINING AND INPUT SAMPLES ---- \\
        if devmode = true then
            stats <- Array.create 10 0
            trainingpatterncount <- 6
            trainingpatternstorage <- Array.create trainingpatterncount "empty"
            classlabelstorage <- Array.create trainingpatterncount "empty"

            //Defining the patterns that are to be stored
            //NEED TO BE SAME LENGTH!
            //Any vector containing only of 0's must have class 0!!! (code restriction)
            trainingpatternstorage.[0] <- "0001"
            classlabelstorage.[0] <- "0"
            trainingpatternstorage.[1] <- "0011"
            classlabelstorage.[1] <- "0"
            trainingpatternstorage.[2] <- "0000"
            classlabelstorage.[2] <- "0"
            trainingpatternstorage.[3] <- "1111"
            classlabelstorage.[3] <- "1"
            trainingpatternstorage.[4] <- "1000"
            classlabelstorage.[4] <- "1"
            trainingpatternstorage.[5] <- "1100"
            classlabelstorage.[5] <- "1"
            inputpatternstorage.[0] <- "0010"

        else
            Console.Write("\n=========== USER INPUT =========== \n")
            // Ask the user for the number of patterns
            Console.Write("Number of trainingspatterns to store: ")

            // Read user input
            trainingpatterncount <- int (Console.ReadLine())

            trainingpatternstorage <- Array.create trainingpatterncount "empty"
            stats <- Array.create trainingpatterncount 0
            classlabelstorage <- Array.create trainingpatterncount "empty"

            Console.Write("\nNOTE: \nAll trainingspattern must have the same length! \nAny vector containing only of 0's must have class 0! (code restriction) \n")

            for c in 0..trainingpatterncount-1 do
                // Ask the user for pattern
                Console.Write("\nPattern {0}: ", (c+1))
                // Read user input
                trainingpatternstorage.[c] <- Console.ReadLine()

                //Retrieve class for current pattern
                Console.Write("Class (0 or 1) of Pattern {0}: ", (c+1))
                classlabelstorage.[c] <- Console.ReadLine()

            // Ask the user for pattern
            Console.Write("\nInputpattern to be classified: ")
            // Read user input
            inputpatternstorage.[0] <- Console.ReadLine()

        show ""
        show "=========== START ==========="
        show ""

        // ---- STRING MANIPULATION IF TEST MODE IS ON ---- \\
        if test = true then
            //FOR TESTING!
            //produce the expected outcomes
            expect <- Array.create trainingpatterncount "empty"
            for k in 0..trainingpatterncount-1 do
                expect.[k] <- inputpatternstorage.[0] + classlabelstorage.[k] + "0" + trainingpatternstorage.[k]


        // ---- COMPUTING THE PATTERN LENGTH AND NUMBER OF REQUIRED QUBITS ---- \\

        //find the length of the patterns
        let trainingpatternlength = trainingpatternstorage.[0].Length
        show "Binary pattern length: %i" trainingpatternlength

        //memory register (patternlength long); loading register (patternlength long); utility register (2 qubits)
        let requiredqubits = 2*trainingpatternlength+2
        show "Number of required qubits: %i" requiredqubits
        let k = Ket(requiredqubits) //initialize state vector
        let psi = k.Qubits //create qubit list

        //initialize controlqubits array (needed for class qubit flipping)
        let controlqubitpositions = Array.create trainingpatternlength 0

        //Save important positionmarkers
        let u1position = trainingpatternlength
        let u2position = trainingpatternlength + 1
        let memoryregisterstart = u2position + 1


        // ---- MAIN LOOP ---- \\
        for i in 0..runs-1 do

            show "Run Nr. %i" i

            //show "ITERATION %i" i
            let psi = k.Reset() //reset the state vector



            // ---- STATE PREPARATION (see Trugenberger, 2001) ---- \\

            X   [psi.[u2position]] //flip the second utility qubit

            for p in 1..trainingpatterncount do
                StorageAlgorithm trainingpatternstorage trainingpatternlength p psi

            show "Initial quantum superposition prepared."

            // ---- MEMORY REGISTER IS NOW PREPARED ---- \\



            // ---- QUANTUM KNN ALGORITHM (see Schuld et al., 2014) ---- \\

            show "Classification algorithm initialized."

            // TO KEEP TRACK OF WHAT'S HAPPENING
            // trainings register --> all qubits from psi.[memoryregisterstart] are part of the initialized memory register superposition
            // input register --> the first register from psi.[0] until psi.[u1position-1]
            // u1 is going to become the class qubit
            // u2 is going to be used as the ancilla qubit

            //Rename and redefine the position variables for the KNN algorithm
            let classqubitposition = u1position
            let ancillaqubitposition = u2position
            let inputregisterend = u1position - 1

            // ---- MODIFYING THE FIRST REGISTER ---- \\

            //Loading the inputvector
            for h in 0..inputregisterend do
                //compare the inputpattern with the last trainingpattern (which is still loaded in the first register)
                if inputpatternstorage.[0].[h] <> trainingpatternstorage.[trainingpatterncount-1].[h] then
                    X   [psi.[h]] //flip the qubit if appropriate

            // ---- FLIPPING THE CLASS LABEL ---- \\
            //since all class labels are currently 0!

            for k in 0..trainingpatterncount-1 do
                let savepos = Array.create (trainingpatternlength) 0
                //only need to flip when class label should be 1
                if classlabelstorage.[k] = "1" then
                    //let mutable poscounter = 0
                    for m in 0..trainingpatternlength-1 do
                        if trainingpatternstorage.[k].[m] = '0' then
                            X   [psi.[memoryregisterstart+m]]
                            savepos.[m] <- 1
                        controlqubitpositions.[m] <- m
                    nCNOTforClasses trainingpatternlength memoryregisterstart controlqubitpositions classqubitposition psi

                    //apply NOT again to all the previously flipped qubits (reverse the previous action)
                    for m in 0..trainingpatternlength-1 do
                        if savepos.[m] = 1 then
                            X   [psi.[memoryregisterstart+m]]

                    //restore the controlqubitpositions to 0
                    for r in 0..trainingpatternlength-1 do
                        controlqubitpositions.[r] <- 0

            // ---- RUNNING THE HEART OF THE KNN ALGORITHM ---- \\
            if test = false then
                SchuldQMLAlg inputregisterend memoryregisterstart classqubitposition ancillaqubitposition psi

            // ---- RETRIEVING STATS ---- \\

            if test = false then
                //retrieve the ancilla stats
                let w = psi.[ancillaqubitposition].Bit.v
                stats.[w] <- stats.[w] + 1

                //Conditional measurement (CM)
                if w = 0 then
                    //if CM was successful measure the class qubit
                    M [psi.[classqubitposition]]
                    let c = psi.[classqubitposition].Bit.v
                    cstats.[c] <- cstats.[c] + 1
                    //count the number of successful CMs for statistics later on
                    conditionalcounter <- conditionalcounter + 1
            else
                //collectstats requiredqubits patternstorage stats psi
                collectteststats stats psi
        // ---- CALCULATE THEORETICAL PREDICTIONS (see Schuld et al., 2014) ---- \\

        hd <- Array.create trainingpatterncount 0.
        let mutable probancpart2 = 0.
        let mutable probclass0part2 = 0.
        let mutable probclass1part2 = 0.

        //Calculate Hamming distances between input and training vectors
        for s in 0..trainingpatterncount-1 do
            for m in 0..trainingpatternlength-1 do
                if trainingpatternstorage.[s].[m] <> inputpatternstorage.[0].[m] then
                    hd.[s] <- hd.[s] + 1.
            probancpart2 <- probancpart2 + ((cos (System.Math.PI/(2.*(float trainingpatternlength))*hd.[s]))**2.)
            if classlabelstorage.[s] = "0" then
                probclass0part2 <- probclass0part2 + ((cos (System.Math.PI/(2.*(float trainingpatternlength))*hd.[s]))**2.)
            else
                probclass1part2 <- probclass1part2 + ((cos (System.Math.PI/(2.*(float trainingpatternlength))*hd.[s]))**2.)

        let probanc = (1./(float trainingpatterncount))*probancpart2
        let probclass0 = (1./((float trainingpatterncount)*probanc))*probclass0part2
        let probclass1 = (1./((float trainingpatterncount)*probanc))*probclass1part2


        // ---- OUTPUT RESULTS ---- \\

        if test = false then
          show ""
          show "=========== STATS ==========="
          show ""
          show "Ancilla measured as 0: %d" stats.[0]
          show "Ancilla measured as 1: %d" stats.[1]
          show "Prediction: %f" probanc
          show ""
          show "Class measured as 0: %d" cstats.[0]
          show "Experimental prob class 0: %f" ((float cstats.[0])/(float conditionalcounter))
          show "Prediction class 0: %f" probclass0
          show ""
          show "Class measured as 1: %d" cstats.[1]
          show "Experimental prob class 1: %f" ((float cstats.[1])/(float conditionalcounter))
          show "Prediction class 1: %f" probclass1

          show ""
          show "=========== CLASSIFICATION ==========="
          show ""
          if cstats.[0] > cstats.[1] then
            show "Input classified as |0>"
          else
            show "Input classified as |1>"
        else

        // ---- TESTSTATS FOR DEBUGGING ---- \\

          show "========= STATS ========="

          //let expect = Array.create trainingpatterncount "empty"
          for t in 0..trainingpatterncount-1 do
              show "expected outcome %i: " (t+1)
              show "|%s>" expect.[t]
              show "count: %i" stats.[t]

        (*show "Measured |01000011>: %i" stats.[0]
        show "Measured |01010101>: %i" stats.[1]
        show "Measured |01100100>: %i" stats.[2]
        show "Measured |01110111>: %i" stats.[3]
        show "Measured |1001001111>: %i" stats.[4]
        show "Measured |1001101011>: %i" stats.[5]
        show "Measured |1001101101>: %i" stats.[6]*)

        (*show "Measured |10101000>: %i" stats.[4]
        show "Measured |11100111>: %i" stats.[5]
        show "Measured |11100011>: %i" stats.[6]
        show "Measured |11100101>: %i" stats.[7]*)


module ParsingWindowDiffusion =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates----\\

    let DiffusionGateOriginal (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGateOriginal", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGateOriginal",
                    Help = "Diffusion operator as originally proposed by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (1.-delta),0.);(0,1,sqrt delta,0.);(1,0,sqrt delta,0.);(1,1,-sqrt (1.-delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let DiffusionGateAdapted (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGateAdapted", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGateAdapted",
                    Help = "Diffusion operator as intended by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (delta),0.);(0,1,sqrt (1.-delta),0.);(1,0,sqrt (1.-delta),0.);(1,1,-sqrt (delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let DiffusionGateAdapted2 (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGateAdapted2", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGateAdapted2",
                    Help = "Diffusion operator as intended by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (delta),0.);(0,1,sqrt (1.-delta),0.);(1,0,sqrt (1.-delta),0.);(1,1,-sqrt (delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    //-----END: define new gates----\\

    //-----START: define new functions----\\

    let collectteststats (stats:_[]) (qs:Qubits) =

        M >< qs
        let a,b,c = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v
        //show "qubits: %i %i %i" a b c

        match a,b,c with
                | 0,0,0 -> stats.[0] <- stats.[0] + 1.0
                | 1,0,0-> stats.[1] <- stats.[1] + 1.0
                | 0,1,0 -> stats.[2] <- stats.[2] + 1.0
                | 0,0,1 -> stats.[3] <- stats.[3] + 1.0
                | 1,1,0 -> stats.[4] <- stats.[4] + 1.0
                | 0,1,1 -> stats.[5] <- stats.[5] + 1.0
                | 1,0,1 -> stats.[6] <- stats.[6] + 1.0
                | 1,1,1 -> stats.[7] <- stats.[7] + 1.0
                | _,_,_ -> show "error" //to handle all other cases

    let collectteststats2 (stats:_[]) (qs:Qubits) =

        M >< qs
        let a,b,c,d = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v
        //show "qubits: %i %i %i" a b c

        match a,b,c,d with
                | 0,0,0,0 -> stats.[0] <- stats.[0] + 1.0
                | 1,0,0,0 -> stats.[1] <- stats.[1] + 1.0
                | 0,1,0,0 -> stats.[2] <- stats.[2] + 1.0
                | 0,0,1,0 -> stats.[3] <- stats.[3] + 1.0
                | 0,0,0,1 -> stats.[4] <- stats.[4] + 1.0
                | 1,1,0,0 -> stats.[5] <- stats.[5] + 1.0
                | 1,0,1,0 -> stats.[6] <- stats.[6] + 1.0
                | 1,0,0,1 -> stats.[7] <- stats.[7] + 1.0
                | 0,1,1,0 -> stats.[8] <- stats.[8] + 1.0
                | 0,1,0,1 -> stats.[9] <- stats.[9] + 1.0
                | 0,0,1,1 -> stats.[10] <- stats.[10] + 1.0
                | 1,0,1,1 -> stats.[11] <- stats.[11] + 1.0
                | 1,1,1,0 -> stats.[12] <- stats.[12] + 1.0
                | 1,1,0,1 -> stats.[13] <- stats.[13] + 1.0
                | 0,1,1,1 -> stats.[14] <- stats.[14] + 1.0
                | 1,1,1,1 -> stats.[15] <- stats.[15] + 1.0
                | _,_,_,_ -> show "error" //to handle all other cases

    //-----END: define new functions----\\

    [<LQD>]
    let __ParsingWindowDiffusion() =

        //Choose number of qubits, number of runs and diffusion delta value
        let qubitnumber = 3
        let runs = 100000
        let delta = 0.7

        //initialize the boolean for stats output
        let mutable three = true

        if qubitnumber <> 3 then
            three <- false

        //Initialize stats array
        let statsize = 2.**(float qubitnumber)
        let stats = Array.create (int statsize) 0.

        //Initialize a  qubit state
        let k = Ket(qubitnumber)
        let qs = k.Qubits

        //MAIN LOOP
        for m in 0..runs-1 do
            let qs = k.Reset()

            // 3 qubit case: prepare the state |010>
            // 4 qubit case: prepare the state |0100>
            X   [qs.[1]]
            X   [qs.[2]]
            //X   [qs.[0]]
            //X   [qs.[2]]
            //X   [qs.[3]]

            // Apply the diffusion operator to all qubits
            for i in 0..qubitnumber-1 do
                DiffusionGateAdapted   delta   [qs.[i]]
            // Apply the diffusion operator to all qubits
            //for i in 0..qubitnumber-1 do
                //DiffusionGateAdapted2   0.7   [qs.[i]]

            if three = true then
                // Retrieve the output
                collectteststats stats qs
            else
                // Retrieve the output
                collectteststats2 stats qs

        let floatruns = float (runs)
        if three = true then

            show "-----------------------------"
            show "---------- RESULTS ----------"
            show "With sqrt(d) and -sqrt(d) on the diagonal"
            show "Measured |000>: %f" (stats.[0]/(floatruns))
            show "Measured |001>: %f" (stats.[3]/(floatruns))
            show "Measured |010>: %f" (stats.[2]/(floatruns))
            show "Measured |011>: %f" (stats.[5]/(floatruns))
            show "Measured |100>: %f" (stats.[1]/(floatruns))
            show "Measured |101>: %f" (stats.[6]/(floatruns))
            show "Measured |110>: %f" (stats.[4]/(floatruns))
            show "Measured |111>: %f" (stats.[7]/(floatruns))
        else
            show "-----------------------------"
            show "---------- RESULTS ----------"
            show "With sqrt(d) and -sqrt(d) on the diagonal"
            show "Measured |0000>: %f" (stats.[0]/(floatruns))
            show "Measured |1000>: %f" (stats.[1]/(floatruns))
            show "Measured |0100>: %f" (stats.[2]/(floatruns))
            show "Measured |0010>: %f" (stats.[3]/(floatruns))
            show "Measured |0001>: %f" (stats.[4]/(floatruns))
            show "Measured |1100>: %f" (stats.[5]/(floatruns))
            show "Measured |1010>: %f" (stats.[6]/(floatruns))
            show "Measured |1001>: %f" (stats.[7]/(floatruns))
            show "Measured |0110>: %f" (stats.[8]/(floatruns))
            show "Measured |0101>: %f" (stats.[9]/(floatruns))
            show "Measured |0011>: %f" (stats.[10]/(floatruns))
            show "Measured |1011>: %f" (stats.[11]/(floatruns))
            show "Measured |1110>: %f" (stats.[12]/(floatruns))
            show "Measured |1101>: %f" (stats.[13]/(floatruns))
            show "Measured |0111>: %f" (stats.[14]/(floatruns))
            show "Measured |1111>: %f" (stats.[15]/(floatruns))

        (*
        let stats = Array.create 8 0.
        for m in 0..runs-1 do

            let qs = k.Reset()

            // prepare the state |010>
            X   [qs.[1]]

            // Apply the diffusion operator to all qubits
            for i in 0..2 do
                DiffusionGateOriginal   delta   [qs.[i]]


            // Retrieve the output
            collectteststats stats qs

        show "-----------------------------"
        show "---------- RESULTS ----------"
        show "With sqrt(1-d) and -sqrt(1-d) on the diagonal"
        show "Measured |000>: %f" (stats.[0]/(floatruns))
        show "Measured |100>: %f" (stats.[1]/(floatruns))
        show "Measured |010>: %f" (stats.[2]/(floatruns))
        show "Measured |001>: %f" (stats.[3]/(floatruns))
        show "Measured |110>: %f" (stats.[4]/(floatruns))
        show "Measured |011>: %f" (stats.[5]/(floatruns))
        show "Measured |101>: %f" (stats.[6]/(floatruns))
        show "Measured |111>: %f" (stats.[7]/(floatruns))
        *)

module diffusionKNN =
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates----\\

    let InverseDiffusionGate (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("InverseDiffusionGate*", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "InverseDiffusionGate*",
                    Help = "Inverse Diffusion operator as originally proposed by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (1.-delta),0.);(0,1,sqrt delta,0.);(1,0,sqrt delta,0.);(1,1,-sqrt (1.-delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let DiffusionGate (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGate", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGate",
                    Help = "Diffusion operator as intended by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (delta),0.);(0,1,sqrt (1.-delta),0.);(1,0,sqrt (1.-delta),0.);(1,1,-sqrt (delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let DiffusionGate2 (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGate2", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGate2",
                    Help = "Diffusion operator as intended by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (delta),0.);(0,1,sqrt (1.-delta),0.);(1,0,sqrt (1.-delta),0.);(1,1,-sqrt (delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    let DiffusionGateAdapted2 (delta:float) (qs:Qubits) =
        let gate =
            Gate.Build("DiffusionGateAdapted2", fun () ->
                new Gate(
                    //Qubits = qs.Length,
                    Name = "DiffusionGateAdapted2",
                    Help = "Diffusion operator as intended by Maria Schuld",
                    Mat  = (CSMat(2,[(0,0,sqrt (delta),0.);(0,1,sqrt (1.-delta),0.);(1,0,sqrt (1.-delta),0.);(1,1,-sqrt (delta),0.)]))
                    //Draw = .....,
            ))
        gate.Run qs

    //-----END: define new gates----\\

    //-----START: define new functions----\\

    let collectteststats (stats:_[]) (qs:Qubits) =

        M >< qs
        let a,b,c = qs.[0].Bit.v, qs.[1].Bit.v, qs.[2].Bit.v
        //show "qubits: %i %i %i" a b c

        match a,b,c with
                | 0,0,0 -> stats.[0] <- stats.[0] + 1.0
                | 1,0,0-> stats.[1] <- stats.[1] + 1.0
                | 0,1,0 -> stats.[2] <- stats.[2] + 1.0
                | 0,0,1 -> stats.[3] <- stats.[3] + 1.0
                | 1,1,0 -> stats.[4] <- stats.[4] + 1.0
                | 0,1,1 -> stats.[5] <- stats.[5] + 1.0
                | 1,0,1 -> stats.[6] <- stats.[6] + 1.0
                | 1,1,1 -> stats.[7] <- stats.[7] + 1.0
                | _,_,_ -> show "error" //to handle all other cases

    let collectteststats2 (stats:_[]) (qs:Qubits) =


        let a,b,c,d = qs.[1].Bit.v, qs.[2].Bit.v, qs.[3].Bit.v, qs.[4].Bit.v
        //show "qubits: %i %i %i" a b c

        match a,b,c,d with
                | 0,0,0,0 -> stats.[0] <- stats.[0] + 1.0
                | 1,0,0,0 -> stats.[1] <- stats.[1] + 1.0
                | 0,1,0,0 -> stats.[2] <- stats.[2] + 1.0
                | 0,0,1,0 -> stats.[3] <- stats.[3] + 1.0
                | 0,0,0,1 -> stats.[4] <- stats.[4] + 1.0
                | 1,1,0,0 -> stats.[5] <- stats.[5] + 1.0
                | 1,0,1,0 -> stats.[6] <- stats.[6] + 1.0
                | 1,0,0,1 -> stats.[7] <- stats.[7] + 1.0
                | 0,1,1,0 -> stats.[8] <- stats.[8] + 1.0
                | 0,1,0,1 -> stats.[9] <- stats.[9] + 1.0
                | 0,0,1,1 -> stats.[10] <- stats.[10] + 1.0
                | 1,0,1,1 -> stats.[11] <- stats.[11] + 1.0
                | 1,1,1,0 -> stats.[12] <- stats.[12] + 1.0
                | 1,1,0,1 -> stats.[13] <- stats.[13] + 1.0
                | 0,1,1,1 -> stats.[14] <- stats.[14] + 1.0
                | 1,1,1,1 -> stats.[15] <- stats.[15] + 1.0
                | _,_,_,_ -> show "error" //to handle all other cases

    //-----END: define new functions----\\

    [<LQD>]
    let __DiffusionKNN() =

        //Choose number of qubits, number of runs and diffusion delta value
        let diffusionqubitnumber = 4
        let qubitnumber = 3+diffusionqubitnumber
        let runs = 10000
        let delta = 0.9
        //let delta1 = 0.6
        //let delta2 = 0.8

        //initialize the boolean for stats output
        let mutable conditionalcounter = 0

        //Initialize cstats array
        let cstats = Array.create 2 0.

        //Initialize stats array
        let statsize = 2.**(float qubitnumber)
        let stats = Array.create (int statsize) 0.

        //Initialize a  qubit state
        let k = Ket(qubitnumber)
        let qs = k.Qubits

        // ------ INFO: QUBIT STATE SETUP ----\\
        // |1 ancilla qubit; 4 qubits for diffusion; 1 class qubit; 1 m-qubit>
        let ancillaqubitpos = 0
        let diffusionregisterstart = 1
        let diffusionregisterend = diffusionqubitnumber
        let classqubitpos = qubitnumber-2
        let mqubitpos = qubitnumber-1

        //MAIN LOOP
        for m in 0..runs-1 do

            let qs = k.Reset()

            // ---- STATE PREPARATION ---- \\

            //put the m-qubit into superposition
            H   [qs.[mqubitpos]]

            //put the ancilla qubit into superposition
            H   [qs.[ancillaqubitpos]]

            // 4 qubits for diffusion remain in the state |0000> for the input vector preparation

            //|1100>
            //CNOT [qs.[ancillaqubitpos];qs.[1]]
            //CNOT [qs.[ancillaqubitpos];qs.[2]]

            // Apply the diffusion operator to all qubits in the diffusion register
            // controlled by ancilla qubit
            for i in 1..diffusionregisterend do
                Cgate (DiffusionGate   delta)   [qs.[ancillaqubitpos];qs.[i]]
            
            //flip the ancilla and move the input vector onto the |0> ancilla state
            X   [qs.[ancillaqubitpos]]

            //prepare the |0100> state in the diffusion register
            //only when ancilla is |1>
            CNOT [qs.[ancillaqubitpos];qs.[2]]

            // Apply the diffusion operator to all qubits in the diffusion register
            // controlled by ancilla and m register!
            for i in 1..diffusionregisterend do
                Cgate (Cgate (DiffusionGate   delta))   [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[i]]
            
            //correct the minus sign
            Cgate (Cgate Z) [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[2]]

            //flip the m qubit and move the trainingsvector 1 to the |0> state of the m qubit
            X   [qs.[mqubitpos]]

            //prepare the |1111> state in the diffusion register
            //only when ancilla and m qubit are both |1>
            //Cgate CNOT [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[2]]
            Cgate CNOT [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[3]]
            Cgate CNOT [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[4]]
            Cgate CNOT [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[1]]

            for i in 1..diffusionregisterend do
                Cgate (Cgate (DiffusionGate   delta))   [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[i]]
                //Cgate (Cgate (InverseDiffusionGate   delta))   [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[i]]
            
            //correct the minus signs
            Cgate (Cgate Z) [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[1]]
            Cgate (Cgate Z) [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[2]]
            Cgate (Cgate Z) [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[3]]
            Cgate (Cgate Z) [qs.[ancillaqubitpos];qs.[mqubitpos];qs.[4]]

            //flip the class label when the m qubit is |1>
            CNOT [qs.[mqubitpos];qs.[classqubitpos]]

            //--- THE QML ALGORITHM ---\\

            //Hadamard on ancilla interferes trainings and input vectors
            H   [qs.[ancillaqubitpos]]

            //Measure the ancilla
            M   [qs.[ancillaqubitpos]]

            //if ancilla was found in |0> state
            if qs.[ancillaqubitpos].Bit.v = 0 then
                    //measure class qubit
                    M   [qs.[classqubitpos]]
                    if qs.[classqubitpos].Bit.v = 0 then
                        cstats.[0] <- cstats.[0]+1.0
                    else
                        cstats.[1] <- cstats.[1]+1.0
                    conditionalcounter <- conditionalcounter + 1
            
        let floatruns = float (runs)
        //let floatruns = float (conditionalcounter)
        show "-----------------------------"
        show "---------- RESULTS ----------"
        show " "
        show "Ancilla measured in |0> state: %f" ((float conditionalcounter)/floatruns)
        show "Ancilla measured in |1> state: %f" ((floatruns-(float conditionalcounter))/floatruns)
        show "-----------------------------"
        show "Class qubit measured in |0> state: %f" (cstats.[0]/(float conditionalcounter))
        show "Class qubit measured in |1> state: %f" (cstats.[1]/(float conditionalcounter))


        show "-----------------------------"
        show "---------- RESULTS ----------"
        show "With sqrt(d) and -sqrt(d) on the diagonal"
        show "Measured |0000>: %f" (stats.[0]/(floatruns))
        show "Measured |1000>: %f" (stats.[1]/(floatruns))
        show "Measured |0100>: %f" (stats.[2]/(floatruns))
        show "Measured |0010>: %f" (stats.[3]/(floatruns))
        show "Measured |0001>: %f" (stats.[4]/(floatruns))
        show "Measured |1100>: %f" (stats.[5]/(floatruns))
        show "Measured |1010>: %f" (stats.[6]/(floatruns))
        show "Measured |1001>: %f" (stats.[7]/(floatruns))
        show "Measured |0110>: %f" (stats.[8]/(floatruns))
        show "Measured |0101>: %f" (stats.[9]/(floatruns))
        show "Measured |0011>: %f" (stats.[10]/(floatruns))
        show "Measured |1011>: %f" (stats.[11]/(floatruns))
        show "Measured |1110>: %f" (stats.[12]/(floatruns))
        show "Measured |1101>: %f" (stats.[13]/(floatruns))
        show "Measured |0111>: %f" (stats.[14]/(floatruns))
        show "Measured |1111>: %f" (stats.[15]/(floatruns))

   
module GroverStatePreparation = 
    open System
    open Util
    open Operations
    //open Native             // Support for Native Interop
    //open HamiltonianGates   // Extra gates for doing Hamiltonian simulations
    //open Tests              // All the built-in tests

    //-----START: define new gates----\\

    //-----END: define new gates----\\

    //-----START: define new functions----\\

    //-----END: define new functions----\\


    [<LQD>]
    let __GroverStatePreparation() = 
        // Algorithm described in Grover & Rudolph (2002) "Creating superpositions that correspond to efficiently integrable probability distributions"

        // First try:
        // GAUSSIAN with mean=0 and std=1
        let qubitnumber = 2
        let k = Ket(qubitnumber)
        let qs = k.Qubits

        //Initialize the Hadamard state since the gaussian distribution is symmetric around the mean
        // 50% in region 0 (left of the mean) and 50% in region 1 (right of the mean)
        H   [qs.[0]]




module Main =
    open App

    /// <summary>
    /// The main entry point for Liquid.
    /// </summary>
    [<EntryPoint>]
    let Main _ =
        RunLiquid ()
