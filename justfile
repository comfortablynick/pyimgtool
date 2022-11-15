bin_name := 'pyimgtool'
test_file_in := 'test/sunset.jpg'
test_file_out := 'test/sunset_edited.jpg'
test_file_out2 := 'test/sunset_edited2.jpg'
test_watermark_file := 'test/logo.png'
test_full_watermark_file := 'test/logo_full.png'

alias r := runargs
alias run := runcv
alias h := help
alias build := runcv # in case vim calls it
alias install := runcv # in case vim calls it

# Run with optional args
runargs +args='':
	{{bin_name}} {{test_file_in}} {{test_file_out}} {{args}}

# test
runpil:
    {{bin_name}} -vv open {{test_file_in}} resize -s 0.4 save {{test_file_out}} -fk

runcv:
    {{bin_name}} -vv open2 {{test_file_in}} resize2 -s 0.4 save {{test_file_out2}} -fk

# test text watermark
runw:
    {{bin_name}} -vv open2 {{test_file_in}} resize2 -W 2000 -H 2000 text2 "Nick Murphy | murphpix.com" -c save {{test_file_out2}} -fk

# test logo watermark image
runwi:
    {{bin_name}} -vv open {{test_file_in}} resize -W 2000 -H 2000 watermark {{test_watermark_file}} -s 1 -o 0.3 save {{test_file_out}} -fk

# test logo watermark image2
runwi2:
    {{bin_name}} -vv open2 {{test_file_in}} resize2 -W 2000 -H 2000 watermark2 {{test_watermark_file}} -s 1 -o 0.3 save {{test_file_out2}} -fk

# test full logo watermark
runwif:
    {{bin_name}} -vv open2 {{test_file_in}} resize2 -W 2000 -H 2000 watermark2 {{test_full_watermark_file}} -s 0.2 -o 0.2 save {{test_file_out2}} -fk

# show prog help
help:
    {{bin_name}} -h

# create dist and upload to pypi
pack:
    rm -rf dist && python setup.py sdist bdist_wheel && twine upload dist/*

# update .tasks file in project root with justfile content
update_tasks:
    just2tasks ./.tasks
