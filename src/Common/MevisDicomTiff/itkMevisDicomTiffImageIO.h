/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMevisDicomTiffImageIO.h,v $
  Language:  C++
  Date:      $Date: 2009/10/14 13:28:12 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMevisDicomTiffImageIO_h
#define __itkMevisDicomTiffImageIO_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include <fstream>
#include <string>

#include "itkImageIOBase.h"
#include "itk_tiff.h"

namespace itk
{


/** \class MevisDicomTiffImageIO
 *
 *  ImageIO for handling Mevis dcm/tiff images,
 *
 *  PROPERTIES:
 *  - developed using gdcm 2.0.10, tiff 3.8.2 and itk 3.10.0
 *  - only 2D/3D, scalar types supported
 *  - input/output tiff image expected to be tiled
 *  - types supported uchar, char, ushort, short, uint, int, and float
 *    (double is not accepted by MevisLab)
 *  - writing defaults is tiled tiff, tilesize is 128, 128, 
 *    LZW compression and cm metric system
 *  - default extension for tiff-image is ".tif" to comply with mevislab
 *    standards
 *
 *  GDCM (current 2.0.12):
 *  - gdcm header during reading is stored as (global) metadata
 *    to fill in the gdcm header when writing. All 'image' values,
 *    eg voxelsize, dimensions etc are overwritten, except
 *    min/max value of the image and intercept/slope.
 *
 *  TIFF (current 3.9.1):
 *
 *  ITK (current 3.14.0):
 
 *  FUNCTIONALITIES:
 *  - reading gdcm file 
 *    always 3D to allocate memory for spacing, dimensions etc
 *    gdcm::DataSet header, storing header into metadict
 *    using attribute to read standard values, except min/max
 *    (somehow is not supported by gdcm). The superclass may
 *    resize the variables depending on the dimensions (eg spacing,
 *    dimensions etc). Therefore when reading we check whether the 
 *    image is 2d or 3d based on dcm header file, and do a re-sizing
 *    of the vector if required. 
 *  - writing gdcm file 
 *    fixed adding comments to see which version has been used
 *    pixeltype of dcm header is unsigned short for int, float 
 *    and double images (see bugfix 20 feb 09)
 *
 *  todo
 *  - streaming support, rgb support
 *  - inch support for tiff
 *  - user selection of compression
 *  - implementing writing tiffimages if x,y < 16 (tilesize)
 *  - add uniform testing for linux, windows, 32/64 bits
 *    in particular for reading/creating/writing dcm files 
 *    and all header values, as well 2d/3d. Things to consider
 *    for testing 1. proper handling position 2. proper handling
 *    pixeltype and sign 3. windows/linux testing 4. both dcm
 *    and tiff (especially dcm file handling)
 *
 *  20 Feb 2009
 *    bugfixed; always set the pixeltype of the dcm image to
 *    unsigned short when writing, otherwise the origin is not
 *    read in correctly by mevislab (for int, float, double)
 *  30 sep 2009
 *    bugfix: consistent handling of 2d/3d throughout code,
 *    thanks to Stefan Klein for pointing out of this bug which
 *    revealed after usage on 2d on windows and thanks for
 *    his suggestions to fix this.
 * 
 *
 *  email: rashindra@gmail.com
 *
 *  \ingroup IOFilters
 */

class TIFFReaderInternal;

class ITK_EXPORT MevisDicomTiffImageIO : public ImageIOBase
{
public:

  typedef MevisDicomTiffImageIO         Self;
  typedef ImageIOBase                   Superclass;
  typedef SmartPointer<Self>            Pointer;
  
  itkNewMacro(Self);
  itkTypeMacro(MevisDicomTiffImageIO, Superclass);
  itkGetMacro(RescaleSlope, double);
  itkGetMacro(RescaleIntercept, double);

  virtual bool CanReadFile(const char*);
  virtual void ReadImageInformation();
  virtual void Read(void* buffer);
  virtual bool CanWriteFile(const char*);
  virtual void WriteImageInformation();
  virtual void Write(const void* buffer);
  virtual bool CanStreamRead()
    {
    return false;
    }

  virtual bool CanStreamWrite()
    {
    return false;
    }

protected:
  MevisDicomTiffImageIO();
  ~MevisDicomTiffImageIO();
  void PrintSelf(std::ostream& os, Indent indent) const;
  
private:
  
  MevisDicomTiffImageIO(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // the following includes the pathname
  // (if these are given)!
  std::string                           m_DcmFileName;
  std::string                           m_TiffFileName;

  TIFF *                                m_TIFFImage;
  unsigned int                          m_TIFFDimension;
  bool                                  m_IsOpen;
  unsigned short                        m_Compression;
  unsigned int                          m_BitsPerSample;
  unsigned int                          m_Width;
  unsigned int                          m_Length;
  unsigned int                          m_Depth;
  bool                                  m_IsTiled;
  unsigned int                          m_TileWidth;
  unsigned int                          m_TileLength;
  unsigned int                          m_TileDepth;
  unsigned short                        m_NumberOfTiles;

  double                                m_RescaleSlope;
  double                                m_RescaleIntercept;
  double                                m_EstimatedMinimum;
  double                                m_EstimatedMaximum;


};

} // end namespace itk

#endif // __itkMevisDicomTiffImageIO_h
