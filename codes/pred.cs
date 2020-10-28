using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class pred : MonoBehaviour
{
    public NNModel modelAsset;
    private Model model;
    
    void Start()
    {
        model = ModelLoader.Load(modelAsset);
        IWorker worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharp, model);  
        Tensor input = new Tensor(0, 3, new float[] {-0.301426f,0.715417f,0.214615f});  
        worker.Execute(input);
        var res = worker.PeekOutput();
        Debug.Log(res[0]);
        input.Dispose();
    }
}

// angle: 21.129271
// pred:  19.843548
