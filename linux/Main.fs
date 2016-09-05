namespace Microsoft.Research.Liquid

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

    //define a quatum function
    let qfunc (qs:Qubits) =
        //Hadamard gate
        //H   qs

        //rotate qubit by 90 degrees in x direction (Math.PI/2. different direction than H rotation)
        rotX (Math.PI/4.)   qs //do this to first qubit

        //ENTANGLEMENT
        for q in qs.Tail do CNOT [qs.Head;q]//select the other qubits but the first one
        //apply a CNOT between the head qubit and each other qubit >> results in entanglement

        //M does measurement of one qubit (it does the first one in the list)
        M >< qs //bow tie operator applies measurement to all qubits

    [<LQD>] //means it can be called from the command line
    let __UserSample(n:int) =
        //gather statistics
        //array with 2 elements and initialize with 0 >> measuring a qubit and collecting the stats
        let stats   = Array.create 2 0
        //create state vector containing a single qubit
        let k       = Ket(n)
        //create circuit
        let circ    = Circuit.Compile qfunc k.Qubits

        //Information retrieval

        show "test1:"
        //output the circuit into the log file
        circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        circ.RenderHT("Test1")

        //convolutes the quantum gates >> impossible on an actual quantum computer
        //but leads to a speed up in the classical simulation
        //e.g. collapsing 9 CNOT gates into one matrix >> speed up!
        let circ    = circ.GrowGates(k)

        show "test2:"
        //output the circuit into the log file
        circ.Dump()
        //Draw it into HTML (H) and Tex (T)
        circ.RenderHT("Test2")


        for i in 0..9999 do
            //reset the state vector since the measurement will collapse the state vector
            let qs  = k.Reset(n)
            //instead of 'qfunc qs' I can run the circuit which does nothing else than running qfunc
            circ.Run qs
            //retrieve the bit value of the first qubit in list qs and convert to integer (v)
            let v   = qs.Head.Bit.v
            //dot between stats and brackets is needed when retrieving objects from a list or array
            stats.[v] <- stats.[v] + 1

            //Test for entanglement
            for q in qs.Tail do
                if q.Bit <> qs.Head.Bit then
                    failwith "BAD!!!!!"

        show "Measured: 0-%d 1-%d" stats.[0] stats.[1]

module Main =
    open App

    /// <summary>
    /// The main entry point for Liquid.
    /// </summary>
    [<EntryPoint>]
    let Main _ =
        RunLiquid ()
