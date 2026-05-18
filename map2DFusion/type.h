// #ifndef __ZY_TYPE_H__
// #define __ZY_TYPE_H__

#include <stdio.h>
#include <stdint.h>
#include <string>
#include <sstream>
#include <math.h>


namespace pi{
typedef unsigned char   byte;

template <class Type,int Size>
struct Array_
{
    Array_(){}
    Array_(Type def){
        for(int i=0;i<Size;i++)
            data[i]=def;
    }
    Type data[Size];

    inline friend std::ostream& operator <<(std::ostream& os,const Array_<Type,Size>& p)
    {
        for(int i=0;i<Size;i++)
            os<<p.data[i]<<" ";
        return os;
    }

    inline friend std::istream& operator >>(std::istream& os,const Array_<Type,Size>& p)
    {
        for(int i=0;i<Size;i++)
            os>>p.data[i];
        return os;
    }

    const int size(){return Size;}
};

}
// #endif // __RTK_TYPE_H__