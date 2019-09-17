bin_name := 'pyimg.py'

alias r := runargs
alias run := runtest
alias h := help
alias build := runtest # in case vim calls it
alias install := runtest # in case vim calls it

# Run with optional args
runargs +args='':
	python {{bin_name}} {{args}}

# test
runtest:
    ./{{bin_name}} test/sunset.jpg test/sunset_edited.jpg -vv -mw 2000

# test watermark image
runwi:
    ./{{bin_name}} test/sunset.jpg test/sunset_edited.jpg -vv -mw 2000 -mh 2000 -wi test/logo.png

# test watermark image (tinify)
runwit:
    ./{{bin_name}} test/sunset.jpg test/sunset_edited.jpg -vv -mw 2000 -mh 2000 -wi test/logo.png -t

# show prog help
help:
    ./{{bin_name}} -h
