/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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
 *  - first public version (no 4D support)
 *      developed using gdcm 2.0.10, tiff 3.8.2 and itk 3.10.0
 *
 *  PROPERTIES:
 *  - 2D/3D/4D, scalar types supported
 *  - input/output tiff image expected to be tiled
 *  - types supported uchar, char, ushort, short, uint, int, and float
 *    (double is not accepted by MevisLab)
 *  - writing defaults is tiled tiff, tilesize is 128, 128,
 *    LZW compression and cm metric system
 *  - default extension for tiff-image is ".tif" to comply with mevislab
 *    standards
 *  - gdcm header during reading is stored as (global) metadata
 *    to fill in the gdcm header when writing. All 'image' values,
 *    eg voxelsize, dimensions etc are overwritten, except
 *    min/max value of the image and intercept/slope.
 *  - note: when reading and writing dcm/tiff 2D files, the third dimension
 *    of spacing and origin are replaced with default values! In MevisLab
 *    these vars are contained.
 *  - BUG in ML: the x/y spacing of the tiff file is swapped with respect
 *    to the x/y spacing in dcm file (dcm info is used btw)
 *
 *  NOTES:
 *  - if a 2d dcm/tiff file from a 3d dataset (e.g. one slice
 *    of a patient) is converted, then the third value for spacing
 *    and position are lost (this is the way itk works, while in
 *    dcm file these values are defined)
 *  - tiff image is always 2D or 3D
 *
 *  todo
 *  - implementing writing tiffimages if x,y < 16 (tilesize)
 *  - adding gantry tilt to test data!
 *  - replacing messages using itkExceptions
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
 *  11 dec 2010
 *    added 4d support, note tiff image is always 2D or 3D
 *
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
  itkGetMacro(GantryTilt, double);

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
  double                                m_GantryTilt;
  double                                m_EstimatedMinimum;
  double                                m_EstimatedMaximum;


};

} // end namespace itk

#endif // __itkMevisDicomTiffImageIO_h
