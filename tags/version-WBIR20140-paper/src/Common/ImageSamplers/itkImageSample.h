/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageSample_h
#define __ImageSample_h

#include "itkNumericTraits.h"

namespace itk
{

  /** \class ImageSample
   *
   * \brief A class that defines an image sample, which is
   * the coordinates of a point and its value.
   *
   */

  template < class TImage >
  class ImageSample
  {
  public:

    //ImageSample():m_ImageValue(0.0){};
    ImageSample(){};
    ~ImageSample() {};

    /** Typedef's. */
    typedef TImage                                      ImageType;
    typedef typename ImageType::PointType               PointType;
    typedef typename ImageType::PixelType               PixelType;
    typedef typename NumericTraits<PixelType>::RealType RealType;

    /** Member variables. */
    PointType   m_ImageCoordinates;
    RealType    m_ImageValue;
  }; // end class ImageSample


} // end namespace itk

#endif // end #ifndef __ImageSample_h

