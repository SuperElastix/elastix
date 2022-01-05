/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkMevisDicomTiffImageIO_h
#define itkMevisDicomTiffImageIO_h

#ifdef _MSC_VER
#  pragma warning(disable : 4786)
#endif

#include "itkImageIOBase.h"
#include "itk_tiff.h"
#include "gdcmTag.h"
#include "gdcmAttribute.h"

#include <fstream>
#include <string>

namespace itk
{

/** \class MevisDicomTiffImageIO
 *
 *  ImageIO for handling Mevis dcm/tiff images,
 *  - first public version (no 4D support)
 *      developed using gdcm 2.0.10, tiff 3.8.2 and itk 3.10.0
 *
 *  NOTES:
 *  - It is good to realize that ITK images have some properties like spacing
 *    and orientatin defined only for the dimensionality of the image,
 *    e.g. 2D slices does not have 3D vector sizes of origin/spacing etc., while
 *    dicom may have these defined.
 *    The other way around, 4D information are not stored in the dicom file,
 *    e.g. origin[3] will be lost when writing. When reading we use default
 *    values for origin (=0) and spacing (=1). Another important issue is the
 *    direction matrix, dicom only stores x/y vectors, and calculates the third
 *    one using outer vector product (assuming right-hand coordinate system).
 *  - tiff image is always 2D or 3D
 *  - IMPORTANT: tiff has been designed for max 32 bits addressable memory block,
 *    the use of 64 bits has been specified in bigtiff.org, and is supported
 *    from libtiff version 4.0 and above. Unfortunately, ML does not
 *    support bigtiff, and probably will not in the future. Therefore, this
 *    class remain as is as long ML will not upgrade, only bug fixes will
 *    be considered
 *  - IMPORTANT: 4d information is lost in the writing of the dicom files!
 *    This severely limits the use of this format in the future. Default values
 *    for 4D spacing is set at 1.0 and for origin at 0.0 (when reading)

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
 *  18 apr 2011
 *    added reading dicom tags from sequences of tags, suggestion and
 *    code proposal by Reinhard Hameeteman
 *
 *  email: rashindra@gmail.com
 *
 *  \ingroup IOFilters
 */

class TIFFReaderInternal;

class ITK_EXPORT MevisDicomTiffImageIO : public ImageIOBase
{
public:
  using Self = MevisDicomTiffImageIO;
  using Superclass = ImageIOBase;
  using Pointer = SmartPointer<Self>;

  itkNewMacro(Self);
  itkTypeMacro(MevisDicomTiffImageIO, Superclass);
  itkGetConstMacro(RescaleSlope, double);
  itkGetConstMacro(RescaleIntercept, double);
  itkGetConstMacro(GantryTilt, double);

  virtual bool
  CanReadFile(const char *);

  virtual void
  ReadImageInformation();

  virtual void
  Read(void * buffer);

  virtual bool
  CanWriteFile(const char *);

  virtual void
  WriteImageInformation();

  virtual void
  Write(const void * buffer);

  virtual bool
  CanStreamRead()
  {
    return false;
  }


  virtual bool
  CanStreamWrite()
  {
    return false;
  }


protected:
  MevisDicomTiffImageIO();
  ~MevisDicomTiffImageIO();
  void
  PrintSelf(std::ostream & os, Indent indent) const;

private:
  MevisDicomTiffImageIO(const Self &);
  void
  operator=(const Self &);

  bool
  FindElement(const gdcm::DataSet ds, const gdcm::Tag tag, gdcm::DataElement & de, const bool breadthfirstsearch);

  // the following may include the pathname
  std::string m_DcmFileName;
  std::string m_TiffFileName;

  TIFF *         m_TIFFImage;
  unsigned int   m_TIFFDimension;
  bool           m_IsOpen;
  unsigned short m_Compression;
  unsigned int   m_BitsPerSample;
  unsigned int   m_Width;
  unsigned int   m_Length;
  unsigned int   m_Depth;
  bool           m_IsTiled;
  unsigned int   m_TileWidth;
  unsigned int   m_TileLength;
  unsigned int   m_TileDepth;
  unsigned short m_NumberOfTiles;

  double m_RescaleSlope;
  double m_RescaleIntercept;
  double m_GantryTilt;
  double m_EstimatedMinimum;
  double m_EstimatedMaximum;
};

} // end namespace itk

#endif // itkMevisDicomTiffImageIO_h
