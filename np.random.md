#### numpy.random.rand()
- return array between [0, 1)
```python
np.random.rand(4,3,2) # shape 4*3*2

```

####  numpy.random.randn()
- return array with standard normal distribution

```python
np.random.randn() # return one number
-1.1241580894939212
```

#### numpy.random.randint()
- numpy.random.randint(low, high=None, size=None, dtype=’l’)
- return random integers from low (inclusive) to high (exclusive)
```python
np.random.randint(1,size=5) # return 0 array[0, 0, 0, 0, 0])

np.random.randint(1,5) # 4
```

#### numpy.random.random_integers
- return random integers from low low (inclusive) to high (inclusive)

#### Generate float between [0,1)

numpy.random.random_sample(size=None)
numpy.random.random(size=None)
numpy.random.ranf(size=None)

#### numpy.random.choice()
- return a random sample from a given 1-D array
```python
np.random.choice(5,3) # array([4, 1, 4])
np.random.choice(5, 3, replace=False) # no repeat number
```

#### numpy.random.seed()
```python
numpy.random.sample(size=None)
```
