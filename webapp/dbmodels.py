from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Item(Base):
    __tablename__ = "items_on_sale"

    id              = Column(String, primary_key=True)
    title           = Column(String)
    price           = Column(Integer)
    pred_price      = Column(Integer)
    sub_category_1  = Column(String)
    sub_category_2  = Column(String)
    brand           = Column(String)
    status          = Column(String)
    shipping        = Column(String)
    description     = Column(String)
    url             = Column(String)
    img_url         = Column(String)
    region          = Column(String)
    period          = Column(String)