import re

text_1 = '''Lorem ipsum dolor
sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista
egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta
lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.'''


re.findall('#[a-zA-Z]*',text_1)
