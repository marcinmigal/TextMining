import re

# Przyklad a)

text_1 = 'Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku'

re.sub('\d ','',text_1)



# Przyklad b)

text_2 = '<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>'

re.sub('<[^>]*>', '', text_2)

# Przyklad c)

text_3 = '''Lorem ipsum dolor sit amet, consectetur; adipiscing elit.
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. In
blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risus
eu risus.'''

re.sub('[^a-zA-Z^ ]',' ', text_3)
