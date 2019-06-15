#### super()
- super([type[, object-or-type]])
- Return a proxy object that delegates method calls to a parent or sibling class of type. This is 
useful for accessing inherited methods that have been overridden in a class.
```python
class A:
     def add(self, x):
         y = x+1
         print(y)
class B(A):
    def add(self, x):
        super().add(x)
b = B()
b.add(2)  # 3
```

```python
class Bird:
    def __init__(self):
          self.hungry = True
    def eat(self):
          if self.hungry:
               print 'Ahahahah'
          else:
               print 'No thanks!'

class SongBird(Bird):
     def __init__(self):
          self.sound = 'Squawk'
     def sing(self):
          print self.song()

sb = SongBird()
sb.sing()    # 能正常输出
sb.eat()     # 报错，因为 songgird 中没有 hungry 特性
```
To fix using super():
```python
class SongBird(Bird):
     def __init__(self):
          super(SongBird,self).__init__()
          self.sound = 'Squawk'
     def sing(self):
          print self.song()
```
