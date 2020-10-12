using UnityEngine;
using System.Linq;
using System;
using System.Collections.Generic;

public class Neuron
{
    float bias;
    float[] weights;
    Func<float, float> activation;

    public Neuron(float b, float[] w, Func<float,float> f)
    {
        bias = b;
        weights = w;
        activation = f;
    }

    public float propagate(float[] inputs)
    {
        return activation(bias + weights.Zip(inputs, (x, y) => x * y).Aggregate((x, y) => x + y));
    }
}

public class MLP
{
    List<List<Neuron>> network;
    
    public MLP()
    {
        network = new List<List<Neuron>>();
    }

    public void addLayer(List<Neuron> l)
    {
        network.Add(l);
    }

    public float propagate(float[] inputs)
    {
        return Mathf.Round(network.Aggregate(inputs, (i, l) => l.Select(n => n.propagate(i)).ToArray())[0]);
    }
}

public class NNs : MonoBehaviour
{
    void Start()
    {
        Func<float,float> relu = x => Mathf.Max(0f, x);
        Func<float,float> sigm = x => 1 / (1 + Mathf.Exp(-x));

        MLP clf = new MLP();
        List<Neuron> nl = new List<Neuron>();
        nl.Add(new Neuron(0.9916f, new float[] {-2.9073f, -0.2868f}, relu));
        nl.Add(new Neuron(0.3341f, new float[] {0.6037f, -0.9359f}, relu));
        clf.addLayer(nl);
        nl = new List<Neuron>();
        nl.Add(new Neuron(1.2597f, new float[] {-1.9851f, 1.3447f}, sigm));
        clf.addLayer(nl);
        
        Debug.Log(clf.propagate(new float[] {-0.6f, 0.3f})); 
    }
}
