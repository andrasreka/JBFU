import customtkinter
import os
from PIL import Image
from bilateral import jbf
import cv2

# def exp():


#     with open('config.json', 'w') as f:
#         json.dump({
#             'kernel' : wnd.config.get('kernel size'),
#             'sigma_spatial': wnd.config.get('sigma spatial'),
#             'sigma_range' : wnd.config.get('sigma range')
#         }, f)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Fusion GUI")
        self.geometry("2024x1024")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gui_images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "CustomTkinter_logo_single.png")), size=(26, 26))
        self.large_test_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "large_test_image.png")), size=(500, 150))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size=(20, 20))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "home_light.png")), size=(20, 20))
      
        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text=" Menu", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_large_image_label = customtkinter.CTkLabel(self.home_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

         # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Scene 2005")
        self.tabview.add("Other")
        self.tabview.tab("Scene 2005").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Other").grid_columnconfigure(0, weight=1)

        self.image_list_combo = customtkinter.CTkComboBox(self.tabview.tab("Scene 2005"),
                                                    values=['Art','Books', 'Dolls', 'Laundry', 'Moebius', 'Reindeer'])
        self.image_list_combo.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.image_list_combo2 = customtkinter.CTkComboBox(self.tabview.tab("Other"),
                                                    values=['Lena'])
        self.image_list_combo2.grid(row=1, column=0, padx=20, pady=(10, 10))

        # create slider

        self.s1_val = 5
        self.s2_val = 10
        self.s3_val = 10

        self.sidebar_frame = customtkinter.CTkFrame(self.tabview, width=250)
        self.sidebar_frame.grid(row=2, column=0,rowspan=4,  padx=(20, 0), pady=(20, 0))
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.slider_1 = customtkinter.CTkSlider(self.sidebar_frame, command=self.slider1_callback, from_=0, to=10)
        self.slider_1.grid(row = 1, column =0)
        self.slider_1.set(self.s1_val)
        
        self.label_tab_1 = customtkinter.CTkLabel(self.sidebar_frame, text="Kernel Size")
        self.label_tab_1.grid(row=0, column=0, padx=0.1, pady=0.1)


        self.slider_2 = customtkinter.CTkSlider(self.sidebar_frame, command=self.slider2_callback, from_=0, to=100)
        self.slider_2.grid(row = 3, column =0)
        self.slider_2.set(self.s2_val)
        
        self.label_tab_2 = customtkinter.CTkLabel(self.sidebar_frame, text="Spatial Sigma")
        self.label_tab_2.grid(row=2, column=0, padx=0.1, pady=0.1)


        self.slider_3 = customtkinter.CTkSlider(self.sidebar_frame, command=self.slider3_callback, from_=0, to=100)
        self.slider_3.grid(row = 5, column =0,  padx=20, pady=20)
        self.slider_3.set(self.s3_val)
        
        self.label_tab_3 = customtkinter.CTkLabel(self.sidebar_frame, text="Range Sigma")
        self.label_tab_3.grid(row=4, column=0, padx=0.1, pady=0.1)

        self.label4 = customtkinter.CTkLabel(self.sidebar_frame, text="Kernel Size:" + str(self.s1_val))
        self.label4.grid(row=6, column=0, padx=0.1, pady=0.1)

        self.label5 = customtkinter.CTkLabel(self.sidebar_frame, text="Spatial Sigma:" + str(self.s2_val))
        self.label5.grid(row=7, column=0, padx=0.1, pady=0.1)


        self.label6 = customtkinter.CTkLabel(self.sidebar_frame, text="Range Sigma:" + str(self.s3_val))
        self.label6.grid(row=8, column=0, padx=0.1, pady=0.1)

        self.home_frame_button_1 = customtkinter.CTkButton(self.home_frame, text="Filter", image=self.image_icon_image, command = self.run)
        self.home_frame_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.home_frame_button_2 = customtkinter.CTkButton(self.home_frame, text="Upsample", image=self.image_icon_image, command = self.run)
        self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)

        # self.home_frame_button_2 = customtkinter.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="right")
        # self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)
        # self.home_frame_button_3 = customtkinter.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="top")
        # self.home_frame_button_3.grid(row=3, column=0, padx=20, pady=10)
        # self.home_frame_button_4 = customtkinter.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="bottom", anchor="w")
        # self.home_frame_button_4.grid(row=4, column=0, padx=20, pady=10)

    
        # select default frame
        self.select_frame_by_name("home")

    def run(self):
        if self.tabview.get() == "Scene 2005":
            print(self.image_list_combo.get())
        else:
            print(self.image_list_combo2.get())

        img =  Image.open("data/filter/lena.png")
        self.large_test_image.configure(dark_image = img,light_image = img, size = img.size)
        img_run = cv2.imread("data/filter/lena.png", 0)
        #jbf(img_run, img_run, self.s1_val, self.s2_val, self.s3_val)


    def tab1(self, value):
        print(value)

    def slider1_callback(self, value):
        self.s1_val = int(value)
        self.label4.configure(text = int(value))

    def slider2_callback(self, value):
        self.s2_val = value
        self.label5.configure(text =  '%.1f' % value)

    def slider3_callback(self, value):
        self.s3_val = value
        self.label6.configure(text = '%.1f' % value)

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
       
        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        

    def home_button_event(self):
        self.select_frame_by_name("home")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.change_appearance_mode_event('Dark')
    app.mainloop()

