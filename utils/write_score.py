import xlsxwriter
import cv2
import os

class score_statistics():
    def __init__(self,dir_file="score_statastic",list_of_class=[]):
        self.workbook = xlsxwriter.Workbook(dir_file)
        self.cell_format = self.workbook.add_format()
        self.cell_format.set_align('center')
        self.cell_format.set_align('vcenter')
        # cell_format.set_bg_color("c4d9ec")

        self.cell_format2 = self.workbook.add_format()
        self.cell_format2.set_align('center')
        self.cell_format2.set_align('vcenter')
        self.cell_format2.set_text_wrap()
        self.cell_format2.set_bg_color("9fc2e0")

        self.highlight_format = self.workbook.add_format()
        self.highlight_format.set_align('center')
        self.highlight_format.set_align('vcenter')
        self.highlight_format.set_bg_color("red")

        self.Header = ["image_id", "Image", "Groundtruth", "Predict"]
        self.Header.append("Score")
        self.Header.append("Underkill_case")
        self.Header.append("Overkill_case")
        self.Header.append("Unknown_case")


        self.worksheet = self.workbook.add_worksheet("Test")
        self.start_row = 0
        self.start_column = 1
        self.worksheet.write_row(self.start_row, self.start_column, self.Header, self.cell_format)
        self.worksheet.set_column("C:C", 20)
        self.worksheet.set_column("B:B", 35)
    def add_row(self,list_of_data):
        self.start_row += 1
        self.worksheet.set_row(self.start_row, 100)
        list_of_data[5] = list_of_data[6] = list_of_data[7]="FALSE"
        if list_of_data[3]=='Unknown':
            list_of_data[7]='TRUE'
        if list_of_data[2]!=list_of_data[3]:
            if list_of_data[2]=='Pass':
                list_of_data[6]="TRUE"
            if list_of_data[2] == 'Reject':
                list_of_data[5]="TRUE"

        img=cv2.imread(list_of_data[1])
        img=img[192:320, 192:320]
        if os.path.exists(os.path.join(os.getcwd(),'save_image'))==False:
            os.makedirs(os.path.join(os.getcwd(),'save_image'))
        saved_name=os.path.join(os.getcwd(),'save_image',list_of_data[0])
        cv2.imwrite(saved_name,img)

        for index, info in enumerate(list_of_data):
            excel_format = self.highlight_format if list_of_data[5]=='TRUE' else self.cell_format

            if index!=1:
                self.worksheet.write(self.start_row,
                                     index + 1,
                                     list_of_data[index],
                                     excel_format if index == 5 else self.cell_format)

            else:
                self.worksheet.insert_image(self.start_row, index + 1, saved_name,
                                            {'x_scale': 1, 'y_scale': 1, 'x_offset': 5, 'y_offset': 5,
                                             'object_position': 1})
    def close(self):
        header = [{'header': head} for head in self.Header]
        # print(header)
        # print(Header)
        self.worksheet.add_table(0, 1, self.start_row, len(self.Header), {'columns': header})
        self.worksheet.freeze_panes(1, 0)
        self.worksheet.hide_gridlines(2)
        self.workbook.close()
