/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMevisDicomTiffImageIO.cxx,v $
  Language:  C++
  Date:      $Date: 2009/10/14 13:28:12 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkMevisDicomTiffImageIO.h"
#include "itkArray.h"
#include "itkMetaDataObject.h"
#include "itkVersion.h"
#include "itkNumericTraits.h"

// developed using gdcm 2.0 and libtiff 3.8.2
#include "gdcmAttribute.h"
#include "gdcmByteValue.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmUIDGenerator.h"
#include "gdcmImage.h"
#include "gdcmImageReader.h"
#include "gdcmWriter.h"
#include "gdcmReader.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmTag.h"
#include "gdcmVL.h"
#include "gdcmVR.h"
#include "gdcmVersion.h"
#include "gdcmPrinter.h"

#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_cross.h>

#include <itksys/SystemTools.hxx>

namespace itk
{

// constructor
MevisDicomTiffImageIO::MevisDicomTiffImageIO():
    m_DcmFileName(""),
    m_TiffFileName(""),
    m_TIFFImage(NULL),
    m_TIFFDimension(2),
    m_IsOpen(false),
    m_Compression(0),
    m_BitsPerSample(0),
    m_Width(0),
    m_Length(0),
    m_Depth(0),
    m_IsTiled(true), // default ML
    m_TileWidth(0),
    m_TileLength(0),
    m_TileDepth(0),
    m_NumberOfTiles(0),
    m_RescaleSlope(NumericTraits<double>::One),
    m_RescaleIntercept(NumericTraits<double>::Zero),
    m_EstimatedMinimum(NumericTraits<double>::Zero),
    m_EstimatedMaximum(NumericTraits<double>::Zero)
{
  this->SetNumberOfDimensions(3); 
  this->SetFileType(Binary);

  this->AddSupportedWriteExtension(".dcm");
  this->AddSupportedWriteExtension(".tif");
  this->AddSupportedWriteExtension(".tiff");
  this->AddSupportedWriteExtension(".DCM");
  this->AddSupportedWriteExtension(".TIF");
  this->AddSupportedWriteExtension(".TIFF");

  this->AddSupportedReadExtension(".dcm");
  this->AddSupportedReadExtension(".tif");
  this->AddSupportedReadExtension(".tiff");
  this->AddSupportedReadExtension(".DCM");
  this->AddSupportedReadExtension(".TIF");
  this->AddSupportedReadExtension(".TIFF");

} 
// destructor
MevisDicomTiffImageIO::~MevisDicomTiffImageIO()
{
    if (m_IsOpen)
    {
        TIFFClose(m_TIFFImage);
    }
}
// printself
void MevisDicomTiffImageIO::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "DcmFileName      : " << m_DcmFileName << std::endl; 
  os << indent << "TiffFileName     : " << m_TiffFileName << std::endl;
  os << indent << "TIFFDimension    : " << m_TIFFDimension << std::endl;
  os << indent << "IsOpen           : " << m_IsOpen << std::endl;
  os << indent << "Compression      : " << m_Compression << std::endl;
  os << indent << "BitsPerSample    : " << m_BitsPerSample << std::endl;
  os << indent << "Width            : " << m_Width << std::endl;
  os << indent << "Length           : " << m_Length << std::endl;
  os << indent << "Depth            : " << m_Depth << std::endl;
  os << indent << "IsTiled          : " << m_IsTiled << std::endl;
  os << indent << "TileWidth        : " << m_TileWidth << std::endl;
  os << indent << "TileLength       : " << m_TileLength << std::endl;
  os << indent << "TileDepth        : " << m_TileDepth << std::endl;
  os << indent << "NumberOfTiles    : " << m_NumberOfTiles << std::endl;
  os << indent << "RescaleIntercept : " << m_RescaleIntercept << std::endl;
  os << indent << "RescaleSlope     : " << m_RescaleSlope << std::endl;
}
// canreadfile
bool MevisDicomTiffImageIO::CanReadFile( const char* filename ) 
{ 
    // get names
    const std::string fn = filename;
	const std::string basename = itksys::SystemTools::GetFilenameWithoutLastExtension(fn);
	const std::string ext = itksys::SystemTools::GetFilenameLastExtension(fn);
    std::string pathname = itksys::SystemTools::GetFilenamePath(fn).c_str();

    if (!pathname.empty())
    {
        pathname = pathname + "/";
    }

    if ( basename.empty() )
    {
        std::cout << "mevisIO: no filename specified" << std::endl;;
        return false;
    }

    // prevent from reading same basenames in dir but with dcm/tiff
    // extension!
    if (ext != ".dcm" && ext != ".DCM" && 
            ext != ".tif" && ext != ".TIF" &&
            ext != ".tiff" && ext != ".TIFF" )
    {
        return false;
    }

    // dcmfile present?
    const std::string dname = pathname + basename + ".dcm";
    const std::string Dname = pathname + basename + ".DCM";

    const std::string d = itksys::SystemTools::ConvertToOutputPath(dname.c_str()); 
    const std::string D = itksys::SystemTools::ConvertToOutputPath(Dname.c_str()); 

    std::ifstream f(d.c_str(), std::ios::in | std::ios::binary);
    std::ifstream F(D.c_str(), std::ios::in | std::ios::binary);

    if (!f.is_open() && !F.is_open())
    {
        std::cout << "mevisIO: cannot read (corresponding) dcm file" << std::endl;
        return false;
    }
    if (f.is_open())
    {
        m_DcmFileName = d;
        f.close();
    }
    if (F.is_open())
    {
        m_DcmFileName = D;
        F.close();
    }

    // tiff file present?
    std::string tifname("");
    const std::string t1name = pathname + basename + ".tif";
    const std::string t2name = pathname + basename + ".tiff";
    const std::string t3name = pathname + basename + ".TIF";
    const std::string t4name = pathname + basename + ".TIFF";

    std::ifstream t1(t1name.c_str(), std::ios::in | std::ios::binary);
    std::ifstream t2(t2name.c_str(), std::ios::in | std::ios::binary);
    std::ifstream t3(t3name.c_str(), std::ios::in | std::ios::binary);
    std::ifstream t4(t4name.c_str(), std::ios::in | std::ios::binary);

    if (!t1.is_open() && !t2.is_open() && !t3.is_open() && !t4.is_open())
    {
        std::cout << "mevisIO: cannot read (corresponding) tif file" << std::endl;
        return false;
    }
    if (t1.is_open())
    {
        m_TiffFileName = t1name;
        t1.close();
    }
    if (t2.is_open())
    {
        m_TiffFileName = t2name;
        t2.close();
    }
    if (t3.is_open())
    {
        m_TiffFileName = t3name;
        t3.close();
    }
    if (t4.is_open())
    {
        m_TiffFileName = t4name;
        t4.close();
    }

    // checking if dcm is valid dcm
    gdcm::Reader reader;
    reader.SetFileName(m_DcmFileName.c_str());
    if (! reader.Read())
    {
        std::cout << "error opening dcm file " << m_DcmFileName << std::endl;
        return false;
    }
    
    // checking if tiff is valid tif
    m_TIFFImage = TIFFOpen(m_TiffFileName.c_str(), "rc"); // c is disable strip chopping
    if (m_TIFFImage == NULL)
    {
        std::cout << "mevisIO: error opening tif file " << m_TiffFileName << std::endl;
        return false;
    }
    else
    {
        m_IsOpen = true;
        if (!TIFFGetField(m_TIFFImage, TIFFTAG_IMAGEWIDTH, &m_Width))
        {
            std::cout << "mevisIO: error getting IMAGEWIDTH " << std::endl;
        }
        if (!TIFFGetField(m_TIFFImage, TIFFTAG_IMAGELENGTH, &m_Length))
        {
            std::cout << "mevisIO: error getting IMAGELENGTH " << std::endl;
        }
        if (!TIFFGetField(m_TIFFImage, TIFFTAG_IMAGEDEPTH, &m_Depth))
        {
            m_TIFFDimension = 2;
            m_Depth = 0;
        }
        else
        {
            m_TIFFDimension = 3;
        }
        if (!TIFFGetField(m_TIFFImage, TIFFTAG_COMPRESSION, &m_Compression))
        {
            std::cout << "mevisIO: error getting COMPRESSION" << std::endl;
        }

        m_IsTiled = TIFFIsTiled(m_TIFFImage);
        if (m_IsTiled)
        {
            m_NumberOfTiles = TIFFNumberOfTiles(m_TIFFImage);

            if (!TIFFGetField(m_TIFFImage,TIFFTAG_TILEWIDTH,&m_TileWidth))
            {
                std::cout << "mevisIO: error getting TILEWIDTH " << std::endl;
            }
            if (!TIFFGetField(m_TIFFImage,TIFFTAG_TILELENGTH,&m_TileLength))
            {
                std::cout << "mevisIO: error getting TILELength" << std::endl;
            }
            if (!TIFFGetField(m_TIFFImage, TIFFTAG_TILEDEPTH, &m_TileDepth))
            {
                m_TileDepth = 0;
            }
 
        }
    }

    return true;
}
// readimageinformation
void MevisDicomTiffImageIO::ReadImageInformation()
{ 

    // INFO from DICOM
    //
    // note: position may need to be shifted since mevis uses corner
    // as position, while itk uses the center
    //
    // note: if a 3D image is provided, then we need to 
    // read spacingbetweenslices
    //
    // spacing
    // position (origin)
    // direction

    // we DON'T trust info from the image (within the
    // dcm file), since somehow eg spacing does not
    // return the correct values. Also, in case of mevis,
    // the dcmfile does not contain an image, which causes
    // reader.Read() to return an error. 
    //
    // We trust the dcm header information instead
    gdcm::ImageReader reader;
    reader.SetFileName(m_DcmFileName.c_str());
    reader.Read();
    const gdcm::DataSet header = reader.GetFile().GetDataSet();

    // number of frames (we always assume 3D image, otherwise
    // spacing and orientation in third dimension is lost
    // Note that using SetNumberOfDimensions() reset all these
    // vars, but using member functions only set the corresponding
    // variable without side effect
    // If this variable does not exists or equals one then we
    // resize the numberofdimensions to two
    gdcm::Attribute<0x0028,0x0008> atnf;

    bool is2d(false);
    if (!header.GetDataElement(atnf.GetTag()).IsEmpty())
    {
       atnf.SetFromDataElement(header.GetDataElement(atnf.GetTag()));
       if (atnf.GetValue() > 1)
       {
            m_Dimensions.resize(3);
            m_Dimensions[2] = atnf.GetValue();
       }
       else
       {
           is2d = true;
       }
    }
    else
    {
        is2d = true;
    }
    if (is2d)
    {
        m_NumberOfDimensions = 2;
    }


    // dimenions - col
    gdcm::Attribute<0x0028,0x0011> atdc;
    if (!header.GetDataElement(atdc.GetTag()).IsEmpty())
    {
        atdc.SetFromDataElement(header.GetDataElement(atdc.GetTag()));
        m_Dimensions[0] = atdc.GetValue();
    }
    else
    {
        std::cout << "mevisIO: error reading dimensions-row from dcm-file" << std::endl;
    }

    // dimensions - row
    gdcm::Attribute<0x0028,0x0010> atdr;
    if (!header.GetDataElement(atdr.GetTag()).IsEmpty())
    {
        atdr.SetFromDataElement(header.GetDataElement(atdr.GetTag()));
        m_Dimensions[1] = atdr.GetValue();
    }
    else
    {
        std::cout << "mevisIO: error reading dimensions-row from dcm-file" << std::endl;
    }

    // spacing, always 3d vector
    m_Spacing.resize(3);
    gdcm::Attribute<0x0028,0x0030> atps;
    if (!header.GetDataElement(atps.GetTag()).IsEmpty())
    {
        atps.SetFromDataElement(header.GetDataElement(atps.GetTag()));
        m_Spacing[0] = atps.GetValue(0);
        m_Spacing[1] = atps.GetValue(1);
    }
    else
    {
        std::cout << "mevisIO: error reading pixelspacing from dcm-file" << std::endl;
    }

    gdcm::Attribute<0x0018,0x0088> atss;
    if (!header.GetDataElement(atss.GetTag()).IsEmpty())
    {
        atss.SetFromDataElement(header.GetDataElement(atss.GetTag()));
        m_Spacing[2] = atss.GetValue();
    }
    else
    {
        m_Spacing[2] = 0;
        std::cout << "mevisIO: error reading slicespacing from dcm-file" << std::endl;
    }

    // patient position (origin), always 3d vector
    m_Origin.resize(3);
    gdcm::Attribute<0x0020,0x0032> atpp;
    if (!header.GetDataElement(atpp.GetTag()).IsEmpty())
    {
        atpp.SetFromDataElement(header.GetDataElement(atpp.GetTag()));
        m_Origin[0] = atpp.GetValue(0);
        m_Origin[1] = atpp.GetValue(1);
        m_Origin[2] = atpp.GetValue(2);
    }
    else
    {
        std::cout << "mevisIO: error reading patient position (origin) from dcm-file" << std::endl;
    }

    // orientation (image orientation), always 3d vector
    m_Direction.resize(3);
    gdcm::Attribute<0x0020,0x0037> atio;
    if (!header.GetDataElement(atio.GetTag()).IsEmpty())
    {
        atio.SetFromDataElement(header.GetDataElement(atio.GetTag()));
        vnl_vector<double> row(3), col(3);

        row[0] = atio.GetValue(0);
        row[1] = atio.GetValue(1);
        row[2] = atio.GetValue(2);
        col[0] = atio.GetValue(3);
        col[1] = atio.GetValue(4);
        col[2] = atio.GetValue(5);

        vnl_vector<double> slice = vnl_cross_3d(row, col);
        this->SetDirection(0, row);
        this->SetDirection(1, col);
        this->SetDirection(2, slice);
    }
    else
    {
        std::cout << "mevisIO: error reading image orientation from dcm-file" << std::endl;
    }

    // rescale
    gdcm::Attribute<0x0028,0x1052> atri;
    if (!header.GetDataElement(atri.GetTag()).IsEmpty())
    {
        atri.SetFromDataElement(header.GetDataElement(atri.GetTag()));
        m_RescaleIntercept = atri.GetValue();
    }
    else
    {
        m_RescaleIntercept = NumericTraits<double>::Zero ; // default
    }
 
    // slope
    gdcm::Attribute<0x0028,0x1053> atrs;
    if (!header.GetDataElement(atrs.GetTag()).IsEmpty())
    {
        atrs.SetFromDataElement(header.GetDataElement(atrs.GetTag()));
        m_RescaleSlope = atrs.GetValue();
    }
    else
    {
        m_RescaleSlope = NumericTraits<double>::One; // default
    }

    // HEADER in MetaDICT
    // copying the gdcm dictionary to the itk dictionary, organization
    // dcm header
    //
    // DataSet       (==header) contains DataElements 
    // DataElement   an unit of information as defined by a single entry 
    //               in the data dictionary, contains
    //               Tag     (0x000,0x0000)
    //               VL      value length field
    //               VR      value representation field
    //               Value   the value itself
    // Value         is either bytes array 
    //               sequence of items or sequence of fragments
    // SeqOfItems    contains items, item contains again a
    //               DataSet (!). 
    //
    // We simply put the whole header as value in the meta
    // dictionary, then no interpretation is required

    MetaDataDictionary & dic = this->GetMetaDataDictionary();
    const std::string tag("0");
    EncapsulateMetaData<gdcm::DataSet>(dic, tag, header);


    // INFO from TIFF 
    // determine size
    // numberofcomponents
    // data type
    if (m_TIFFImage == NULL)
    {
        std::cout << "mevisIO: error opening file " << m_TiffFileName << std::endl;
        return;

    }

    // sanity checks, dim and sizes
    if ((is2d && m_TIFFDimension != 2) || (!is2d && m_TIFFDimension != 3))
    {
        std::cout << "mevisIO: dcm/tiff dimensions do not correspond!" << std::endl;
    }
    if ( (m_Width != m_Dimensions[0]) || (m_Length != m_Dimensions[1]) ||
          (m_TIFFDimension == 3 && m_Depth != m_Dimensions[2] )) 
    {
        std::cout << "mevisIO: dcm/tiff sizes do not correspond!" << std::endl;
    }

    // format 1 unsigned int
    // format 2 signed int
    // format 3 float
    // format 4 undefined
    // samplesperpixel : number of components per pixel (1 grayscale, 3 rgb)
    // bitspersample: number of bits per component 
    uint16 format, pixel;
    
    if (!TIFFGetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, &format))
    {
        std::cout << "mevisIO: error getting SAMPLEFORMAT" << std::endl;
    }
    if (!TIFFGetField(m_TIFFImage, TIFFTAG_SAMPLESPERPIXEL, &pixel))
    {
        std::cout << "mevisIO: error getting SAMPLESPERPIXEL" << std::endl;
    }
    if (!TIFFGetField(m_TIFFImage, TIFFTAG_BITSPERSAMPLE, &m_BitsPerSample))
    {
        std::cout << "mevisIO: error getting BITSPERSAMPLE" << std::endl;
    }

   // currently we only support grayscale
   if (pixel == 1)
   {
    this->SetPixelType(SCALAR);
    this->SetNumberOfComponents(1);
   }
   else
   {
       std::cout << "mevisIO: currently only support grayscale" << std::endl;
   }

   bool typeassign(false);

   if (m_BitsPerSample <= 8)
   {
       if (format == 1)
       {
           this->SetComponentType(UCHAR);
           typeassign = true;
       }
       if (format == 2)
       {
           this->SetComponentType(CHAR);
           typeassign = true;
       }
   }
   if (m_BitsPerSample == 16)
   {
       if (format == 1)
       {
           this->SetComponentType(USHORT);
           typeassign = true;
       }
       if (format == 2)
       {
           this->SetComponentType(SHORT);
           typeassign = true;
       }
   }
   if (m_BitsPerSample == 32)
   {
       if (format == 1)
       {
           this->SetComponentType(UINT);
           typeassign = true;
       }
       if (format == 2)
       {
           this->SetComponentType(INT);
           typeassign = true;
       }
       if (format == 3)
       {
           this->SetComponentType(FLOAT);
           typeassign = true;
       }
   }
   if (!typeassign)
   {
       std::cout << "mevisIO: unsupported pixeltype " << std::endl;
   }
   // set compression
   // 1 none
   // 2 ccit
   // 5 lzw
   // 32773 packbits
   if (m_Compression == 2 || m_Compression == 5 || m_Compression == 32773)
   {
       m_UseCompression = true;
   }
   if (m_Compression == 1)
   {
       m_UseCompression = false;
   }

} 
// read
void MevisDicomTiffImageIO::Read(void* buffer)
{ 

    // always assume contigous data (PLANARCONFIG =1)
    // image is either tiled or stripped
    //
    // TIFFTileSize     returns size of one tile in bytes
    // TIFFReadTile     reads one tile, returns number of bytes in decoded tile
    //
    // note *buffer goes in scanline order!
    // very inconvenient if the tiff image is tiled, which damned
    // is the case for mevislab images!
    // note buffer is already allocated, according to size!

    short int p;
    if (!TIFFGetField(m_TIFFImage,TIFFTAG_PLANARCONFIG,&p))
    {
        std::cout << "mevisIO: error getting PLANARCONFIG" << std::endl;
    }
    else
    {
        if (p != 1)
        {
            std::cout << "mevisIO: non-contiguous data!" << std::endl;
            return;
        }
    }

    if (m_IsTiled)
    {
        // only works for tile depth == 1 (used by mevislab),
        // therefore in z-direction we do not need to do checking
        // if the volume is multiple of tile.
        if (m_TIFFDimension == 3 && m_TileDepth != 1)
        {
            std::cout << "mevisIO: unsupported tiledepth (should be one)! " << std::endl;
            return;
        }

        // buffer pointer is scanline based (one dimensional array)
        // tile is positioned on x,y,z; we read each tile, and fill
        // the corresponding positions in the onedimensional array
    	unsigned char *vol = reinterpret_cast<unsigned char*>(buffer); 

        const unsigned int tilesize = TIFFTileSize(m_TIFFImage);
        const unsigned int tilerowbytes = TIFFTileRowSize(m_TIFFImage);
        const unsigned int bytespersample = m_BitsPerSample/8;

        unsigned char *tilebuf = static_cast<unsigned char*>(_TIFFmalloc(tilesize));

        //
        // special cases, if the tilexy is larger than or equal to the image 
        // size itself, treat as three separate cases, both are larger/equal,
        // or only direction
        //
        bool tileoversized(false);

        if (m_TileLength >= m_Length || m_TileWidth >= m_Width)
        {
            tileoversized = true;

            // case one both x,y is larger
            if (m_TileLength >= m_Length && m_TileWidth >= m_Width)
            {
                for (unsigned int z0 = 0; z0< (m_TIFFDimension == 3 ? m_Depth :1); z0++)
                {
                    if (TIFFReadTile(m_TIFFImage, tilebuf, 0, 0, z0, 0) < 0)
                    {
                        std::cout << "mevisIO: error reading tile (topleft)" << std::endl;
                        _TIFFfree(tilebuf);
                        return;
                    }
                    else
                    {
                        // do row based copy of tile into volume
                        const unsigned lenx = m_Width;
                        const unsigned leny = m_Length;
                        const unsigned int tilexbytes = lenx * bytespersample;

                        unsigned char * pv = vol;
                        const unsigned int p = z0 * m_Length * m_Width;
                        pv += p * bytespersample;

                        unsigned char * pb = tilebuf;
                        for (unsigned int r = 0; r<leny; ++r)
                        {
                            memcpy(pv,pb,tilexbytes);
                            pv += tilexbytes;
                            pb += tilerowbytes;
                        }
                    }
                }
            } // end case one

            // case two, larger x, smaller y
            if (m_TileWidth >= m_Width && m_TileLength < m_Length)
            {

                const unsigned lenx = m_Width;
                const unsigned int tilexbytes = lenx * bytespersample;

                const bool my = ( m_Length%m_TileLength == 0) ? true : false;
                for (unsigned int z0 = 0; z0 < (m_TIFFDimension == 3 ? m_Depth : 1); z0++) 
                {
                    for (unsigned int y0 = 0; y0 < (my ? m_Length : m_Length-m_TileLength); y0 += m_TileLength)
                    {
                        if (TIFFReadTile(m_TIFFImage, tilebuf, 0, y0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (top image)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {

                            unsigned char * pb = tilebuf;
                            unsigned char * pv = vol;
                            const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width;
                            pv += p * bytespersample;

                            for (unsigned int r = 0; r<m_TileLength; ++r)
                            {
                                memcpy(pv,pb,tilerowbytes);
                                pv += tilexbytes;
                                pb += tilerowbytes;
                            }
                        }
                    }

                    if (!my)
                    {
                        const unsigned leny = m_Length%m_TileLength;
                        const unsigned int y0 = m_Length - leny;

                        if (TIFFReadTile(m_TIFFImage, tilebuf, 0, y0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (strip bottom)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            unsigned char * pb = tilebuf;
                            unsigned char * pv = vol;
                            const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width;
                            pv += p * bytespersample;

                            for (unsigned int r = 0; r<leny; ++r)
                            {
                                memcpy(pv,pb,tilexbytes);
                                pv += tilexbytes;
                                pb += tilerowbytes;
                            }
                        }
                    }
                }

            } // end case two

            // case three, smaller x, larger y
            if (m_TileWidth < m_Width && m_TileLength >= m_Length)
            {

                const unsigned leny = m_Length;
                const bool mx = ( m_Width%m_TileWidth == 0) ? true: false;

                for (unsigned int z0 = 0; z0 < (m_TIFFDimension == 3 ? m_Depth : 1); z0++) 
                {
                    for (unsigned int x0 = 0; x0 < (mx ? m_Width : m_Width-m_TileWidth); x0 += m_TileWidth)
                    {
                        // read tile
                        if (TIFFReadTile(m_TIFFImage, tilebuf, x0, 0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (top image)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            unsigned char * pb = tilebuf;
                            unsigned char * pv = vol;
                            const unsigned int p = z0 * m_Length * m_Width + x0;
                            pv += p * bytespersample;

                            for (unsigned int r = 0; r<leny; ++r)
                            {
                                memcpy(pv,pb,tilerowbytes);
                                pv += m_Width * bytespersample;
                                pb += tilerowbytes;
                            }
                        }
                    }
                    // fill strip right
                    if (!mx)
                    {
                        const unsigned lenx = m_Width%m_TileWidth;
                        const unsigned int x0 = m_Width - lenx;
                        const unsigned int tilexbytes = lenx * bytespersample;

                        if (TIFFReadTile(m_TIFFImage, tilebuf, x0, 0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (strip right)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            unsigned char * pb = tilebuf;
                            unsigned char * pv = vol;
                            const unsigned int p = z0 * m_Length * m_Width + x0; 
                            pv += p * bytespersample;

                            for (unsigned int r = 0; r<leny; ++r)
                            {
                                memcpy(pv,pb,tilexbytes);
                                pv += m_Width * bytespersample;
                                pb += tilerowbytes;
                            }
                        }
                    }
                }
         
            } // end case three

        } // end oversized tile


        //
        // normal case, tile is smaller than image
        //
        if (!tileoversized)
        {
            // is volume direction a multiple of tiledirection?
            const bool mx = ( m_Width%m_TileWidth == 0) ? true: false;
            const bool my = ( m_Length%m_TileLength == 0) ? true : false;

            // fill everything inside ie from topleft
            for (unsigned int z0 = 0; z0 < (m_TIFFDimension == 3 ? m_Depth : 1); z0++) 
            {
                for (unsigned int y0 = 0; y0 < (my ? m_Length : m_Length-m_TileLength); y0 += m_TileLength)
                    for (unsigned int x0 = 0; x0 < (mx ? m_Width : m_Width-m_TileWidth); x0 += m_TileWidth)
                    {
                        // x0,y0,z0 is position of tile in volume, top left corner
                        // pointer to volume (at 0,0,0)
                        unsigned char * pv = vol;

                        // set pointer of volume to y0,x0,z0 position
                        const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                        pv += p * bytespersample;

                        // read tile
                        if (TIFFReadTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (topleft image)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            // do row based copy of tile into volume
                            unsigned char * pb = tilebuf;
                            for (unsigned int r = 0; r<m_TileLength; ++r)
                            {
                                memcpy(pv,pb,tilerowbytes);
                                // move pointers
                                // x remain same, y is one complete vol line
                                // (ie width of image)
                                pv += m_Width * bytespersample;
                                pb += tilerowbytes;
                            }
                        }

                    }
                // fill boundaries
                if (!mx)
                {
                    // x is fixed
                    const unsigned lenx = m_Width%m_TileWidth;
                    const unsigned int x0 = m_Width - lenx;
                    const unsigned int tilexbytes = lenx * bytespersample;

                    for (unsigned int y0 = 0; y0 < (my ? m_Length: m_Length - m_TileLength); y0 += m_TileLength)
                    {
                        unsigned char * pv = vol;
                        const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                        pv += p * bytespersample;

                        if (TIFFReadTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (ydirection)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            unsigned char * pb = tilebuf;
                            for (unsigned int r = 0; r<m_TileLength; ++r)
                            {
                                memcpy(pv,pb,tilexbytes);
                                pv += m_Width * bytespersample;
                                pb += tilerowbytes;
                            }
                        }
                    }
                }
                if (!my)
                {
                    // y is fixed
                    const unsigned leny = m_Length%m_TileLength;
                    const unsigned int y0 = m_Length - leny;

                    for (unsigned int x0 = 0; x0 < (mx ? m_Width : m_Width-m_TileWidth); x0 += m_TileWidth)
                    {
                        unsigned char * pv = vol;
                        const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                        pv += p * bytespersample;

                        if (TIFFReadTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                        {
                            std::cout << "mevisIO: error reading tile (x-direction)" << std::endl;
                            _TIFFfree(tilebuf);
                            return;
                        }
                        else
                        {
                            unsigned char * pb = tilebuf;
                            for (unsigned int r = 0; r<leny; ++r)
                            {
                                memcpy(pv,pb,tilerowbytes);
                                pv += m_Width * bytespersample;
                                pb += tilerowbytes;
                            }
                        }
                    }
                }
                // fill corner bottom
                if (!mx && !my)
                {
                    // x0,y0 is fixed
                    const unsigned lenx = m_Width%m_TileWidth;
                    const unsigned int x0 = m_Width - lenx;
                    const unsigned int tilexbytes = lenx * bytespersample;
                    const unsigned leny = m_Length%m_TileLength;
                    const unsigned int y0 = m_Length - leny;

                    unsigned char * pv = vol;
                    const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                    pv += p * bytespersample;

                    if (TIFFReadTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                    {
                        std::cout << "mevisIO: error reading tile (corner bottom)" << std::endl;
                        _TIFFfree(tilebuf);
                        return;
                    }
                    else
                    {
                        unsigned char * pb = tilebuf;
                        for (unsigned int r = 0; r<leny; ++r)
                        {
                            memcpy(pv,pb,tilexbytes);
                            pv += m_Width * bytespersample;
                            pb += tilerowbytes;
                        }
                    }
                }
            }
        }
        
        _TIFFfree(tilebuf);
    } 
    else
    {
        // if not tiled then img is stripped
        std::cout << "mevisIO: non-tiled dcm/tiff reading not (yet) implemented" << std::endl;
        return;
    }
    return;
} 
// canwritefile
bool MevisDicomTiffImageIO::CanWriteFile( const char * name )
{

    std::string filename = name;
    if (filename == "")
    {
        return false;
    }

    // get basename/extension
    const std::string fn = filename;
	const std::string basename = itksys::SystemTools::GetFilenameWithoutLastExtension(fn);
	const std::string ext = itksys::SystemTools::GetFilenameLastExtension(fn);

    std::string pathname = itksys::SystemTools::GetFilenamePath(fn).c_str();

    if (!pathname.empty())
    {
        pathname = pathname + "/";
    }

    if ( basename.empty() )
    {
        std::cout << "mevisIO: no filename specified" << std::endl;
        return false;
    }

    // expect extension dcm or tif(f)
    if (ext == ".tif" || ext == ".tiff") 
    {
        m_TiffFileName = pathname + basename + ext;
        m_DcmFileName = pathname + basename + ".dcm";
        return true;
    }
    if (ext == ".TIF" || ext == ".TIFF")
    {
        m_TiffFileName = pathname + basename + ext;
        m_DcmFileName = pathname + basename + ".DCM";
        return true;
    }
    if (ext == ".dcm")
    {
        m_TiffFileName = pathname + basename + ".tif";
        m_DcmFileName = pathname + basename + ".dcm";
        return true;
    }
    if (ext == ".DCM" )
    {
        m_TiffFileName = pathname + basename + ".TIF";
        m_DcmFileName = pathname + basename + ".DCM";
        return true;
    }

    return false;
}
// writeimageinformation
void MevisDicomTiffImageIO
::WriteImageInformation(void)
{
}

// write
void MevisDicomTiffImageIO
::Write( const void* buffer) 
{
    if (this->GetNumberOfDimensions() != 2 && this->GetNumberOfDimensions() != 3)
    {
        itkExceptionMacro(<< "mevisIO: dcm/tiff writer only supports 2D/3D"); 
    }

    std::ofstream dcmfile(m_DcmFileName.c_str(), std::ios::out|std::ios::binary);
    if (!dcmfile.is_open())
    {
        std::cout << "mevisIO: error opening dcm file for writing " << m_DcmFileName << std::endl;
    }
    dcmfile.close();

    // DCM - HEADER
    //
    //
    // Always add everything that is available from the metaheader,
    // the following tags are not replaced if they are available
    // but they may have changed during image processing. Be warned!
    // - dicom rescale intercept/slope values
    // - min/max (code below is after setting bits)
    // - photometric (default min-is-black)
    // - sop imagetype (0008,0008) 
    // - sop class uid (0008,0016)
    // - sop instance uid (0008,0018)
    // - study instance uid (0020,000d)
    // - series instance uid (0020,000e)
    //
    // The following are always replaced:
    // - comments
    // - rows, columns, frames
    // - pixelspacing, spacing between slices
    // - orientation, position
    // - samples per pixel (always 1)
    // - bits allocated
    // - bits stored (always nbits)
    // - high bit (always nbits-1)
    // - pixel representation (0 unsigned, 1 signed)
 
    gdcm::Writer writer;
    writer.SetCheckFileMetaInformation(false);
    gdcm::DataSet &header = writer.GetFile().GetDataSet();
     
    MetaDataDictionary &dict = this->GetMetaDataDictionary();

    const bool emptydict(dict.Begin()==dict.End());

	itk::MetaDataDictionary::ConstIterator dictiter;
	for(dictiter = dict.Begin(); dictiter!= dict.End(); ++dictiter)
    {
		const std::string & key = dictiter->first;
        ExposeMetaData<gdcm::DataSet>(dict, key, header);
    }

    // copy from metaheader if exists otherwise replace
     // rescale
    gdcm::Attribute<0x0028,0x1052> atri;
    if (!emptydict && !header.GetDataElement(atri.GetTag()).IsEmpty())
    {
        atri.SetFromDataElement(header.GetDataElement(atri.GetTag()));
    }
    else
    {
        atri.SetValue(m_RescaleIntercept);
    }
    header.Replace(atri.GetAsDataElement());
    // intercept
    gdcm::Attribute<0x0028,0x1053> atrs;
    if (!emptydict && !header.GetDataElement(atrs.GetTag()).IsEmpty())
    {
        atrs.SetFromDataElement(header.GetDataElement(atrs.GetTag()));
    }
    else
    {
        atrs.SetValue(m_RescaleSlope);
    }
    header.Replace(atrs.GetAsDataElement());

    // photometric
    gdcm::Attribute<0x0028,0x0004> atphoto;
    if (!emptydict && !header.GetDataElement(atphoto.GetTag()).IsEmpty())
    {
        atphoto.SetFromDataElement(header.GetDataElement(atphoto.GetTag()));
    }
    else
    {
        // monochrome2 -- low values dark, high values bright
        atphoto.SetValue("MONOCHROME2");
    }
    header.Replace(atphoto.GetAsDataElement());
    // imagetype
    gdcm::Attribute<0x0008,0x0008> atimagetype;
    if (!emptydict && !header.GetDataElement(atimagetype.GetTag()).IsEmpty())
    {
        atimagetype.SetFromDataElement(header.GetDataElement(atimagetype.GetTag()));
    }
    else
    {
        static const gdcm::CSComp values[] = {"DERIVED","SECONDARY"};
        atimagetype.SetValues(values,2);
    }
    header.Replace(atimagetype.GetAsDataElement());
    // sop class uid
    gdcm::UIDGenerator uid;
    uid.SetRoot("1.2.840.10008.5.1.4.1.1.7");
    gdcm::Attribute<0x0008,0x0016> atsopclass;
    if (!emptydict && !header.GetDataElement(atsopclass.GetTag()).IsEmpty())
    {
        atsopclass.SetFromDataElement(header.GetDataElement(atsopclass.GetTag()));
    }
    else
    {
        atsopclass.SetValue(uid.GetRoot());
    }
    header.Replace(atsopclass.GetAsDataElement());
    // sop instance uid
    uid.SetRoot(uid.GetGDCMUID());
    gdcm::Attribute<0x0008,0x0018> atsopinstance;
    if (!emptydict && !header.GetDataElement(atsopinstance.GetTag()).IsEmpty())
    {
        atsopinstance.SetFromDataElement(header.GetDataElement(atsopinstance.GetTag()));
    }
    else
    {
        atsopinstance.SetValue(uid.Generate());
    }
    header.Replace(atsopinstance.GetAsDataElement());
    // study instance uid
    gdcm::Attribute<0x0020,0x000d> atstudy;
    if (!emptydict && !header.GetDataElement(atstudy.GetTag()).IsEmpty())
    {
        atstudy.SetFromDataElement(header.GetDataElement(atstudy.GetTag()));
    }
    else
    {
        atstudy.SetValue(uid.Generate());
    }
    header.Replace(atstudy.GetAsDataElement());
    // series instance uid
    gdcm::Attribute<0x0020,0x000e> atserie;
    if (!emptydict && !header.GetDataElement(atserie.GetTag()).IsEmpty())
    {
        atserie.SetFromDataElement(header.GetDataElement(atserie.GetTag()));
    }
    else
    {
        atserie.SetValue(uid.Generate());
    }
    header.Replace(atserie.GetAsDataElement());



    // following attributes are always replaced
    // comments 
    gdcm::Attribute<0x0020,0x4000> atc;
    const std::string v(Version::GetITKVersion());
    const std::string g(gdcm::Version::GetVersion());
    const std::string c = "MevisIO: ITK " + v + " GDCM " + g;
    if (atc.GetValue().empty())
    {
        atc.SetValue(c);
        header.Replace(atc.GetAsDataElement());
    }

    // dimension - columns
    gdcm::Attribute<0x0028,0x0011> atdc;
    atdc.SetValue(m_Dimensions[0]);
    header.Replace(atdc.GetAsDataElement());

    // dimension - row
    gdcm::Attribute<0x0028,0x0010> atdr;
    atdr.SetValue(m_Dimensions[1]);
    header.Replace(atdr.GetAsDataElement());

    // number of frames
    gdcm::Attribute<0x0028,0x0008> atnf;
    if (this->GetNumberOfDimensions() == 2)
    {
        atnf.SetValue(1);
    }
    else if (this->GetNumberOfDimensions() == 3)
    {
        atnf.SetValue(m_Dimensions[2]);
    }
    header.Replace(atnf.GetAsDataElement());

    // spacing
    gdcm::Attribute<0x0028,0x0030> atps;
    atps.SetValue(m_Spacing[0],0);
    atps.SetValue(m_Spacing[1],1);
    header.Replace(atps.GetAsDataElement());

    // spacing between slices 
    gdcm::Attribute<0x0018,0x0088> atss;
    if (m_Spacing.size() > 2)
    {
        atss.SetValue(m_Spacing[2]);
    }
    else
    {
        atss.SetValue(1.0);
    }
    header.Replace(atss.GetAsDataElement());

    // samples per pixel
    switch(this->GetPixelType())
    {
        case SCALAR:
            {
                // number of components should be one
                if (this->GetNumberOfComponents() != 1)
                {
                    std::cout << "mevisIO: nr of Components should be 1 for SCALAR" << std::endl;
                    return;
                }
                gdcm::Attribute<0x0028,0x0002> atsamples;
                atsamples.SetValue(1);
                header.Replace(atsamples.GetAsDataElement());
            }
            break;
        default:
            std::cout << "mevisIO: only SCALAR pixeltypes supported" << std::endl;
            return;
    }
 
   // bits allocated, stored, high
   // default is always 16 bits, only for pixeltype with are shorter
    unsigned int    nbits(16);
    bool            sign(false);

    switch(this->GetComponentType())
    {
        case UCHAR:
            {
                nbits = 8;
                sign = false;
                m_EstimatedMinimum = 0;
                m_EstimatedMaximum = 255;
            }break;
        case CHAR:
            {
                nbits = 8;
                sign = true;
                m_EstimatedMinimum = -128;
                m_EstimatedMaximum = 127;
            }break;
        case USHORT:
            {
                nbits = 16;
                sign = false;
                m_EstimatedMinimum = 0;
                m_EstimatedMaximum = 4095;
            }break;
        case SHORT:
            {
                nbits = 16;
                sign = true;
                m_EstimatedMinimum = -1024;
                m_EstimatedMaximum = 3095;
            }break;
         case UINT:
            {
                nbits = 16;
                sign = false;
                m_EstimatedMinimum = 0;
                m_EstimatedMaximum = 4095;
            }break;
         case INT:
            {
                nbits = 16;
                sign = true;
                m_EstimatedMinimum = -1024;
                m_EstimatedMaximum = 3095;
            }break;
         case FLOAT:
            {
                nbits = 16;
                sign = true;
                m_EstimatedMinimum = -1024;
                m_EstimatedMaximum = 3095;
            }break;
        default:
            {
                std::cout << "mevisIO: error writing dcm-file unsupported component type" << std::endl;
                return;
            }
    }

    // bits allocated
    gdcm::Attribute<0x0028,0x0100> atbitsalloc;
    atbitsalloc.SetValue(nbits);
    header.Replace(atbitsalloc.GetAsDataElement());

    // bits stored
    gdcm::Attribute<0x0028,0x0101> atbitsstored;
    atbitsstored.SetValue(nbits);
    header.Replace(atbitsstored.GetAsDataElement());

    // high bit
    gdcm::Attribute<0x0028,0x0102> atbitshigh;
    atbitshigh.SetValue(nbits-1);
    header.Replace(atbitshigh.GetAsDataElement());

    // pixelrepresentation (sign)
    gdcm::Attribute<0x0028,0x0103> atpixel;
    if (sign)
    {
        atpixel.SetValue(1);
    }
    else
    {
        atpixel.SetValue(0);
    }
    header.Replace(atpixel.GetAsDataElement());

    // min
    gdcm::Attribute<0x0028,0x0106, gdcm::VR::SS, gdcm::VM::VM1> atmin;
    if (!emptydict && !header.GetDataElement(atmin.GetTag()).IsEmpty())
    {
        atmin.SetFromDataElement(header.GetDataElement(atmin.GetTag()));
        m_EstimatedMinimum = atmin.GetValue();
    }
    else
    {
        atmin.SetValue(m_EstimatedMinimum);
    }
    header.Replace(atmin.GetAsDataElement());

    // max
    gdcm::Attribute<0x0028,0x0107, gdcm::VR::SS, gdcm::VM::VM1> atmax;
    if (!emptydict && !header.GetDataElement(atmax.GetTag()).IsEmpty())
    {
        atmax.SetFromDataElement(header.GetDataElement(atmax.GetTag()));
        m_EstimatedMaximum = atmax.GetValue();
    }
    else
    {
        atmax.SetValue(m_EstimatedMaximum);
    }
    header.Replace(atmax.GetAsDataElement());

    // position (origin) 
    gdcm::Attribute<0x0020,0x0032> atpp;
    atpp.SetValue(m_Origin[0],0);
    atpp.SetValue(m_Origin[1],1);
    if (m_Origin.size() > 2)
    {
        atpp.SetValue(m_Origin[2],2);
    }
    else
    {
        atpp.SetValue(0,2);
    }
    header.Replace(atpp.GetAsDataElement());

    // orientation
    gdcm::Attribute<0x0020,0x0037> atio;
    std::vector<double> row(3),col(3);
    row = this->GetDirection(0);
    col = this->GetDirection(1);

    atio.SetValue(row[0],0);
    atio.SetValue(row[1],1);
    if (row.size() > 2)
    {
        atio.SetValue(row[2],2);
    }
    else
    {
        atio.SetValue(0,2);
    }
    atio.SetValue(col[0],3);
    atio.SetValue(col[1],4);
    if (col.size() > 2)
    {
        atio.SetValue(col[2],5);
    }
    else
    {
        atio.SetValue( 0.0, 5);
    }

    header.Replace(atio.GetAsDataElement());

     
    writer.SetFileName(m_DcmFileName.c_str());
    if (!writer.Write())
    {
        std::cout << "mevisIO: error writing dcm header file" << std::endl;
    }



    //TIFF
    //
    //default   using compression (note that values from
    //          reading an tiff image are not stored, like
    //          dcm header is)
    //default   minisblack
    //default   tiled

    m_TIFFImage = TIFFOpen(m_TiffFileName.c_str(),"w");
    if (!m_TIFFImage)
    {
        itkExceptionMacro(<< "mevisIO: error opening tiff file for writing"); 
    }

    // software comment
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_SOFTWARE,c.c_str()))
    {
        std::cout << "mevisIO: error setting SOFTWARE" << std::endl;
    }

    // set sizes
    m_Width = m_Dimensions[0];
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_IMAGEWIDTH, m_Width))
    {
        std::cout << "mevisIO: error setting IMAGEWIDTH" << std::endl;
    }
    m_Length = m_Dimensions[1];
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_IMAGELENGTH, m_Length))
    {
        std::cout << "mevisIO: error setting IMAGELENGTH" << std::endl;
    }

    // dimensions
    if (m_NumberOfDimensions == 2 || (m_NumberOfDimensions == 3 && m_Dimensions[2] == 1))
    {
        m_TIFFDimension = 2;
        m_Depth = 0;
    }
    else if (m_NumberOfDimensions == 3)
    {
        m_TIFFDimension = 3;
        m_Depth = m_Dimensions[2];
        if (!TIFFSetField(m_TIFFImage, TIFFTAG_IMAGEDEPTH, m_Depth))
        {
            std::cout << "mevisIO: error setting IMAGEDEPTH" << std::endl;
        }
    }
    // photometric (default min-is-black)
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_PHOTOMETRIC,PHOTOMETRIC_MINISBLACK))
    {
        std::cout << "mevisIO: error setting PHOTOMETRIC" << std::endl;
    }
    // minimumn
    if (sign)
    {
        if (!TIFFSetField(m_TIFFImage, TIFFTAG_SMINSAMPLEVALUE,m_EstimatedMinimum))
        {
            std::cout << "mevisIO: error setting SMINSAMPLEVALUE" << std::endl;
        }
    }
    else
    {
        if (!TIFFSetField(m_TIFFImage, TIFFTAG_MINSAMPLEVALUE,static_cast<unsigned int>(m_EstimatedMinimum)))
        {
            std::cout << "mevisIO: error setting MINSAMPLEVALUE" << std::endl;
        }
    }
    // maximum
    if (sign)
    {
        if (!TIFFSetField(m_TIFFImage, TIFFTAG_SMAXSAMPLEVALUE,m_EstimatedMaximum))
        {
            std::cout << "mevisIO: error setting SMAXSAMPLEVALUE" << std::endl;
        }
    }
    else
    {
        if (!TIFFSetField(m_TIFFImage,TIFFTAG_MAXSAMPLEVALUE,static_cast<unsigned int>(m_EstimatedMaximum)))
        {
            std::cout << "mevisIO: error setting MAXSAMPLEVALUE" << std::endl;
        }
    }
    // pixeltype
    switch(this->GetPixelType())
    {
        case SCALAR:
            {
                // number of components should be one
                if (this->GetNumberOfComponents() != 1)
                {
                    std::cout << "mevisIO: nr of Components should be 1 for SCALAR" << std::endl;
                    return;
                }
                if (!TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLESPERPIXEL, 1))
                {
                    std::cout << "mevisIO: error setting SAMPLESPERPIXEL" << std::endl;
                }
            }
            break;
        default:
            std::cout << "mevisIO: only SCALAR pixeltypes supported" << std::endl;
            return;
    }
    // componenttype
    bool suc(false);
    switch(this->GetComponentType())
    {
        case UCHAR:
            {
                m_BitsPerSample = 8;
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 1);
            }break;
        case CHAR:
            {
                m_BitsPerSample = 8;
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 2);
            }break;
        case USHORT:
            {
                m_BitsPerSample = 8 * sizeof(unsigned short);
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 1);
            }break;
        case SHORT:
            {
                m_BitsPerSample = 8 * sizeof(short);
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 2);
            }break;
         case UINT:
            {
                m_BitsPerSample = 8 * sizeof(unsigned int);
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 1);
            }break;
         case INT:
            {
                m_BitsPerSample = 8 * sizeof(int);
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 2);
            }break;
         case FLOAT:
            {
                m_BitsPerSample = 8 * sizeof(float);
                suc = TIFFSetField(m_TIFFImage, TIFFTAG_SAMPLEFORMAT, 3);
            }break;
        default:
            {
                std::cout << "mevisIO: unsupported component type" << std::endl;
                return;
            }
    }
    if (!suc)
    {
        std::cout << "mevisIO: error setting SAMPLEFORMAT" << std::endl;
    }
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_BITSPERSAMPLE, m_BitsPerSample))
    {
        std::cout << "mevisIO: error setting BITSPERSAMPLE " << std::endl;
    }

    // compression, default always using lzw (overriding
    // member values)
    // 1 none
    // 2 ccit
    // 5 lzw
    // 32773 packbits

    if (!TIFFSetField(m_TIFFImage, TIFFTAG_COMPRESSION, 5))
    {
        std::cout << "mevisIO: error setting COMPRESSION" << std::endl;
    }
    // resolution (always assuming cm)
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER))
    {
        std::cout << "mevisIO: error setting RESOLUTIONUNIT" << std::endl;
    }
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_XRESOLUTION, 10.0/m_Spacing[0]))
    {
        std::cout << "mevisIO: error setting XRESOLUTION " << std::endl;
    }
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_YRESOLUTION, 10.0/m_Spacing[1]))
    {
        std::cout << "mevisIO: error setting XRESOLUTION " << std::endl;
    }

    // setting tilespecs
    m_IsTiled = true; // default
    if (m_NumberOfDimensions == 2 || (m_NumberOfDimensions == 3 && m_Dimensions[2] == 1))
    {
        m_TileDepth = 0;
    }
    else
    {
        m_TileDepth = 1;
        if (!TIFFSetField(m_TIFFImage,TIFFTAG_TILEDEPTH,m_TileDepth))
        {
            std::cout << "mevisIO: error setting TILEDEPTH" << std::endl;
        }
    }

    // following function adjust tilesizes accordingly
    // always set the tile smaller than the image size
    // (which usually is a reasonable assumption, since
    // the images we're dealing with are usually large)
    // defaults (multiple of 16)
    m_TileWidth = 128;
    m_TileLength = 128;

    bool smallimg(false);
    if (m_Width < 16)
    {
        smallimg = true;
        m_TileWidth = 16;
    }
    else
    {
        while (m_TileWidth > m_Width)
        {
            m_TileWidth -= 16;

        }
    }
    if (m_Length < 16)
    {
        smallimg = true;
        m_TileLength = 16;
    }
    else
    {
        while (m_TileLength > m_Length)
        {
            m_TileLength -= 16;
        }
    }

    // function below makes the tile sizes a multiple of 16
    // but does not consider imagelength/width, therefore skipped
    // TIFFDefaultTileSize(m_TIFFImage,&m_TileWidth,&m_TileLength);

    if (!TIFFSetField(m_TIFFImage, TIFFTAG_TILEWIDTH, m_TileWidth))
    {
        std::cout << "mevisIO: error setting TILEWIDTH, m_TileWidth" << std::endl;
    }
    if (!TIFFSetField(m_TIFFImage, TIFFTAG_TILELENGTH, m_TileLength))
    {
        std::cout << "mevisIO: error setting TILELENGTH, m_TileLength" << std::endl;
    }


    // now filling the image with buffer provided
    // the provided buffer is one dimensional array,
    // we apply the same routines as for reading the image
    // except, no boundary checking is required for writing
    // the tiles. Boundary checking on the input pointer to
    // prevent assessing memblocks outside the array

    if (smallimg)
    {
        // We consider images smaller than 16x16xz as a special
        // case, but selecting tile as layout is really not the best
        // choice! For robustness, should also be implemented, but for
        // now left open.
        std::cout << "mevisIO: image x,y smaller than tilesize (16)! Consider" << std::endl;
        std::cout << "         different layout for tif (eg scanline layout)" << std::endl;
            
        TIFFClose(m_TIFFImage);
        return;
    }
    else
    {
        const unsigned int tilesize = TIFFTileSize(m_TIFFImage);
        const unsigned int tilerowbytes = TIFFTileRowSize(m_TIFFImage);
        const unsigned int bytespersample = m_BitsPerSample/8;

        const unsigned char *vol = reinterpret_cast<const unsigned char*>(buffer); 
        unsigned char *tilebuf = static_cast<unsigned char*>(_TIFFmalloc(tilesize));

        // is volume direction a multiple of tiledirection?
        const bool mx = ( m_Width%m_TileWidth == 0) ? true: false;
        const bool my = ( m_Length%m_TileLength == 0) ? true : false;

        for (unsigned int z0 = 0; z0 < (m_TIFFDimension == 3 ? m_Depth:1); z0++)
        {
            for (unsigned int y0 = 0; y0 < (my ? m_Length : m_Length-m_TileLength); y0 += m_TileLength)
                for (unsigned int x0 = 0; x0 < (mx ? m_Width : m_Width-m_TileWidth); x0 += m_TileWidth)
                {
                    // set bufferpointer to begin of tile
                    const unsigned char * pv = vol;
                    const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                    pv += p * bytespersample;

                    // fill tile 
                    unsigned char * pb = tilebuf;
                    for (unsigned int r=0; r<m_TileLength; ++r)
                    {
                        memcpy(pb,pv,tilerowbytes);
                        pv += m_Width * bytespersample;
                        pb += tilerowbytes;
                    }
                    // write tile
                    if (TIFFWriteTile(m_TIFFImage, tilebuf,x0,y0,z0,0)< 0)
                    {
                        std::cout << "mevisIO: error writing tile " << std::endl;
                        _TIFFfree(tilebuf);
                        TIFFClose(m_TIFFImage);
                        return;
                    }
                }
                // boundaries
            if (!mx)
            {
                // x is fixed
                const unsigned lenx = m_Width%m_TileWidth;
                const unsigned int x0 = m_Width - lenx;
                const unsigned int tilexbytes = lenx * bytespersample;

                for (unsigned int y0 = 0; y0 < (my ? m_Length: m_Length - m_TileLength); y0 += m_TileLength)
                {
                    const unsigned char * pv = vol;
                    const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                    pv += p * bytespersample;

                    unsigned char * pb = tilebuf;
                    memset (pb,0,tilesize);

                    // fill tile
                    for (unsigned int r = 0; r<m_TileLength; ++r)
                    {
                        memcpy(pb,pv,tilexbytes);
                        pv += m_Width * bytespersample;
                        pb += tilerowbytes;
                    }

                    if (TIFFWriteTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                    {
                        std::cout << "mevisIO: error writing tile (ydirection)" << std::endl;
                        _TIFFfree(tilebuf);
                        TIFFClose(m_TIFFImage);
                        return;
                    }
                }
            }
            if (!my)
            {
                const unsigned leny = m_Length%m_TileLength;
                const unsigned int y0 = m_Length - leny;

                for (unsigned int x0 = 0; x0 < (mx ? m_Width : m_Width-m_TileWidth); x0 += m_TileWidth)
                {
                    const unsigned char * pv = vol;
                    const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                    pv += p * bytespersample;

                    unsigned char * pb = tilebuf;
                    memset (pb,0,tilesize);

                    for (unsigned int r = 0; r<leny; ++r)
                    {
                        memcpy(pb,pv,tilerowbytes);
                        pv += m_Width * bytespersample;
                        pb += tilerowbytes;
                    }

                    if (TIFFWriteTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                    {
                        std::cout << "mevisIO: error writing tile (x-direction)" << std::endl;
                        _TIFFfree(tilebuf);
                        TIFFClose(m_TIFFImage);
                        return;
                    }
                }
            }
            if (!mx && !my)
            {
                // x0,y0 is fixed
                const unsigned lenx = m_Width%m_TileWidth;
                const unsigned int x0 = m_Width - lenx;
                const unsigned int tilexbytes = lenx * bytespersample;
                const unsigned leny = m_Length%m_TileLength;
                const unsigned int y0 = m_Length - leny;

                const unsigned char * pv = vol;
                const unsigned int p = z0 * m_Length * m_Width + y0 * m_Width + x0;
                pv += p * bytespersample;

                unsigned char * pb = tilebuf;
                memset (pb,0,tilesize);

                for (unsigned int r = 0; r<leny; ++r)
                {
                    memcpy(pb,pv,tilexbytes);
                    pv += m_Width * bytespersample;
                    pb += tilerowbytes;
                }

                if (TIFFWriteTile(m_TIFFImage, tilebuf, x0, y0, z0, 0) < 0)
                {
                    std::cout << "mevisIO: error writing tile (corner bottom)" << std::endl;
                    _TIFFfree(tilebuf);
                    TIFFClose(m_TIFFImage);
                    return;
                }
            }
        } // end z
        _TIFFfree(tilebuf);
    }

    TIFFClose(m_TIFFImage);



    return;

}

} // end namespace itk
