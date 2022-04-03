import matplotlib.pyplot as plt
from wordcloud import WordCloud
from cleaning import text_cleaner
from stemming import stem_text


text_list = []

with open('true.csv', 'r', encoding='utf-8') as file:
    for  i, row  in enumerate(file.readlines()):
        line = text_cleaner(row)
        line = stem_text(line)
        text_list.extend(line)
        # print(i)
        if i >=100:
            break

bow = dict([[x,text_list.count(x)] for x in set(text_list)])





wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()