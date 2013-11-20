/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBinaryANNTreeBase_h
#define __itkBinaryANNTreeBase_h

#include "itkBinaryTreeBase.h"
#include <ANN/ANN.h>  // ANN declarations

namespace itk
{

  /**
   * \class BinaryANNTreeBase
   *
   * \brief
   *
   *
   * \ingroup ANNwrap
   */

  template < class TListSample >
  class BinaryANNTreeBase : public BinaryTreeBase< TListSample >
  {
  public:

    /** Standard itk. */
    typedef BinaryANNTreeBase             Self;
    typedef BinaryTreeBase< TListSample > Superclass;
    typedef SmartPointer< Self >          Pointer;
    typedef SmartPointer< const Self >    ConstPointer;

    /** ITK type info. */
    itkTypeMacro( BinaryANNTreeBase, BinaryTreeBase );

    /** Typedefs from Superclass. */
    typedef typename Superclass::SampleType                 SampleType;
    typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
    typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
    typedef typename Superclass::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;

    /** Typedef */
    typedef ANNpointSet       ANNPointSetType;

    /** Get the ANN tree. */
    virtual ANNPointSetType * GetANNTree( void ) const = 0;

  protected:

    /** Constructor. */
    BinaryANNTreeBase();

    /** Destructor. */
    virtual ~BinaryANNTreeBase() {};

  private:

    BinaryANNTreeBase( const Self& ); // purposely not implemented
    void operator=( const Self& );    // purposely not implemented

  }; // end class BinaryANNTreeBase


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryANNTreeBase.hxx"
#endif

#endif // end #ifndef __itkBinaryANNTreeBase_h
