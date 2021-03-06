<h2>AI Procurement Bot</h2>
<h3>Intro</h3>
<ul>
  <li>Script converts human's language into numbers.</li>
  <li>Each occurance of a word is being tracked down into a list which finally serves to train the ML model.</li>
  <li>Script takes advantage of tensorflow library that is commonly used for deep learning projects.</li>
</ul>

<h3>Demo</h3>
<p>Here is the short demo case of interaction with the bot.</p>
<img src="images/bot.JPG">

## LancasterStemmer
 ```
   >>> from nltk.stem.lancaster import LancasterStemmer
        >>> st = LancasterStemmer()
        >>> st.stem('maximum')     # Remove "-um" when word is intact
        'maxim'
        >>> st.stem('presumably')  # Don't remove "-um" when word is not intact
        'presum'
        >>> st.stem('multiply')    # No action taken if word ends with "-ply"
        'multiply'
        >>> st.stem('provision')   # Replace "-sion" with "-j" to trigger "j" set of rules
        'provid'
        >>> st.stem('owed')        # Word starting with vowel must contain at least 2 letters
        'ow'
        >>> st.stem('ear')         # ditto
        'ear'
        >>> st.stem('saying')      # Words starting with consonant must contain at least 3
        'say'
        >>> st.stem('crying')      #     letters and one of those letters must be a vowel
```

## Extending vs Appending to Python list
```
# Appending
x = [1, 2, 3]
x.append([4, 5])
print (x)
[1, 2, 3, [4, 5]]

# Extending
x = [1, 2, 3]
x.extend([4, 5])
print (x)
[1, 2, 3, 4, 5]
```

## Returning index of specific value in a list
```
list_numbers = [3, 1, 2, 3, 3, 4, 5, 6, 3, 7, 8, 9, 10]
element = 3
list_numbers.index(element)
0
```
