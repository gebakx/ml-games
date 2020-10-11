using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SimpleJSON;
using System.IO;
using System.Linq;
using System;

// requirement: http://wiki.unity3d.com/index.php/SimpleJSON

public class gnbjson : MonoBehaviour
{
    void Start()
    {
        string path = "Assets/Resources/tir.json";
        StreamReader reader = new StreamReader(path); 
        var model = JSON.Parse(reader.ReadToEnd());
        reader.Close();        

        int[] test1 = {14,2,18};
        predict(test1, model);
        int[] test2 = {20,4,11};
        predict(test2, model);
    }

    void predict(int[] test, JSONNode model)
    {
       Func<int, int, double> probFeat = 
            (cl, f) => Math.Exp(-Math.Pow(test[f]-model["theta"][cl][f], 2)/(2*model["sigma"][cl][f])) /
                        Math.Sqrt(2 * Math.PI * model["sigma"][cl][f]);
        Func<int, double> prob = 
            cl => model["class_prior"][cl] * Enumerable.Range(0,test.Length).Select(
                f => probFeat(cl, f)
                ).Aggregate(
                    (x, y) => x * y);

        var pred = Enumerable.Range(0,model["classes"].Count).Select(
            cl => (prob(cl), model["classes"][cl])
            ).Max().Item2;
        Debug.Log(pred);
    }

}
