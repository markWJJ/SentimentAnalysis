# -*- coding:utf-8 -*-
import xlrd
import sys
import xlwt
# reload(sys)
# sys.setdefaultencoding("utf-8")

def get_keywords(excel_path=""):
    key_word_set = set()

    data = xlrd.open_workbook(excel_path)
    table = data.sheets()[0]

    text = ""

    nrows = table.nrows 
    ncols = table.ncols 
    
    keywords = {}
    
    for rownum in range(0,nrows):
        tkey = table.cell(rownum, 8).value
        print(rownum,tkey)
        tkey=str(tkey)
        key_word_set.add(tkey)
        keywords[tkey] = 1
    
    print(nrows,len(keywords),keywords)
    return keywords,key_word_set


def CheckIsLockWord(s="", rulemap={},sent_bt_list=[]):
    l4ruleslist = list(rulemap.keys())
    lock = False
    lockrule = ""
    lockruleword = ""

    all_rule_word=[]
    # for arule in l4ruleslist:
    #     aruleItems = arule.split("&")
    #     all_rule_word.extend(aruleItems)
    # if set(all_rule_word)<=set(sent_bt_list):
    #     return False, "", ""
    # else:
    for arule in l4ruleslist:
        # print(arulek,arule)
        aruleItems = arule.split("&")
        # 如果 本体list 不是规则list的子集
        aruleItemschar=[e for e in arule.replace("&","")]
        sent_bt_listchar=[e for e in "".join(sent_bt_list)]
        if not set(aruleItemschar)>=set(sent_bt_listchar):
            continue

        # if not set(aruleItems)>=set(sent_bt_list):
        #     continue

        matchid = 0
        match = True
        for aruletag in aruleItems:
            tmatchid = s.find(aruletag, matchid)
            # print(s,aruletag,tmatchid)
            if tmatchid == -1:
                match = False
                break
            matchid = tmatchid

        if match == True and len(l4ruleslist) > 0:
            lock = True
            lockrule = arule
            lockruleword = arule
            break

        # print(match)

        if lock == True:
            break
    return lock, lockrule, lockruleword


def get_bt(excel_path="./BT_20190718.xlsx"):

    data = xlrd.open_workbook(excel_path)
    table0 = data.sheets()[0]
    table1 = data.sheets()[1]
    table2 = data.sheets()[2]

    text = ""
    
    btwords = {}
    print(table0.nrows,table0.ncols)
    for rownum in range(1,table0.nrows ):
        tkey = table0.cell(rownum, 0).value
        btwords[tkey] = 1
        
    for rownum in range(1,table1.nrows ):
        tkey = table1.cell(rownum, 0).value
        btwords[tkey] = 1
        
    for rownum in range(1,table2.nrows ):
        tkey = table2.cell(rownum, 0).value
        btwords[tkey] = 1
    
    print(len(btwords),btwords)
    return btwords

def getSentBt(sent,btwords):
    sent_bt_list=[]
    btwords_list=[str(e) for e in list(btwords.keys())]
    for e in btwords_list:
        if e in sent:
            sent_bt_list.append(e)
    return sent_bt_list

def CheckXlsx(excel_path='',rulemap={},btwords={}):
    work_sheet=xlrd.open_workbook(excel_path)
    sheet=work_sheet.sheet_by_index(0)

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('0')

    count_dict={}


    for i in range(sheet.nrows):
        sent=sheet.cell_value(i,2)
        label=sheet.cell_value(i,4)
        sent_bt_list=getSentBt(sent,btwords)
        worksheet.write(i,0,sheet.cell_value(i,0))
        worksheet.write(i,1,sheet.cell_value(i,1))
        worksheet.write(i,2,sheet.cell_value(i,2))
        worksheet.write(i,3,sheet.cell_value(i,3))
        worksheet.write(i,4,sheet.cell_value(i,4))
        worksheet.write(i,5,sheet.cell_value(i,5))


        # print(sent,sent_bt_list)
        lock, lockrule, lockruleword=CheckIsLockWord(sent,rulemap,sent_bt_list)
        print(label,sent,sent_bt_list,lockrule,lock)
        worksheet.write(i,6,str(lock))
        worksheet.write(i,7,str(lockrule))
        worksheet.write(i,8,str(sent_bt_list))


        # count_dict[label]=count_dict.get(label,0)+1
        if label=="是A推其他":
            if lock==True:
                count_dict["是A推其他_锁定"]=count_dict.get("是A推其他_锁定",0)+1
            else:
                count_dict["是A推其他_非锁定"]=count_dict.get("是A推其他_非锁定",0)+1

        if label=="是A推模糊":
            if lock == True:
                count_dict["是A推模糊_锁定"] = count_dict.get("是A推模糊_锁定", 0) + 1
            else:
                count_dict["是A推模糊_非锁定"] = count_dict.get("是A推模糊_非锁定", 0) + 1

        if label=="不是A推A":
            if lock == True:
                count_dict["不是A推A_锁定"] = count_dict.get("不是A推A_锁定", 0) + 1
            else:
                count_dict["不是A推A_非锁定"] = count_dict.get("不是A推A_非锁定", 0) + 1

        if label=="正确":
            if lock == True:
                count_dict["正确_锁定"] = count_dict.get("正确_锁定", 0) + 1
            else:
                count_dict["正确_非锁定"] = count_dict.get("正确_非锁定", 0) + 1



    pr_1=float(count_dict["是A推其他_锁定"])/float(count_dict["是A推其他_锁定"]+count_dict["是A推其他_非锁定"])
    pr_2=float(count_dict["是A推模糊_锁定"])/float(count_dict["是A推模糊_锁定"]+count_dict["是A推模糊_非锁定"])
    pr_3=float(count_dict["不是A推A_非锁定"])/float(count_dict["不是A推A_锁定"]+count_dict["不是A推A_非锁定"])
    pr_4=float(count_dict["正确_锁定"])/float(count_dict["正确_锁定"]+count_dict["正确_非锁定"])

    print("是A推其他_正确率：%s  是A推模糊_准确率:%s  不是A推A_准确率：%s  正确_准确率:%s"%(pr_1,pr_2,pr_3,pr_4))

    workbook.save('自动续费问题_output.xls')


btwords=get_bt(excel_path="./BT_20190718.xlsx")
keywords,key_word_set=get_keywords(excel_path="./KW_4000142_20190718.xlsx")
excel_path="./自动续费问题_deal.xls"
CheckXlsx(excel_path,keywords,btwords)




