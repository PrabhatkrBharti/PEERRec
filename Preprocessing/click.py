from pynput.mouse import Controller, Button

mouse = Controller()

while True:
	mouse.click(Button.left, 1)
	print('clicked')

	time.sleep(5)