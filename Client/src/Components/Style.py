from tkinter import font as tkFont
import src.Components.Menu_Bar as Menu_Bar


def font(size):
    font = tkFont.Font(family="Lucida Grande", size=size)
    font.config()
    return font


def auto_resize(size, isWidth=True):
    if isWidth:
        before = size * 10000 / 1920
        after = int(before * Menu_Bar.screen_width / 10000)
    else:
        before = size * 10000 / 1080
        after = int(before * Menu_Bar.screen_height / 10000)
    return after


