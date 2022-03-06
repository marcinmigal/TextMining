import re

text_1 = '''Lorem ipsum dolor :) sit amet, consectetur; adipiscing elit. Sed eget mattis sem. ;)
Mauris ;( egestas erat quam, :< ut faucibus eros congue :> et. In blandit, mi eu porta;
lobortis, tortor :-) nisl facilisis leo, at ;< tristique augue risus eu risus ;-).'''

re.findall('[:;][)(<>]|[:;][-][)(<>]', text_1)