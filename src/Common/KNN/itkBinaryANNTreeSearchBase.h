/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBinaryANNTreeSearchBase_h
#define __itkBinaryANNTreeSearchBase_h

#include "itkBinaryTreeSearchBase.h"
#include "itkBinaryANNTreeBase.h"
#include "ANN/ANN.h"

namespace itk
{

/**
 * \class BinaryANNTreeSearchBase
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template< class TListSample >
class BinaryANNTreeSearchBase :
  public BinaryTreeSearchBase< TListSample >
{
public:

  /** Standard itk. */
  typedef BinaryANNTreeSearchBase Self;
  typedef BinaryTreeSearchBase<
    TListSample >                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** ITK type info. */
  itkTypeMacro( BinaryANNTreeSearchBase, BinaryTreeSearchBase );

  /** Typedefs from Superclass. */
  typedef typename Superclass::ListSampleType        ListSampleType;
  typedef typename Superclass::BinaryTreeType        BinaryTreeType;
  typedef typename Superclass::BinaryTreePointer     BinaryTreePointer;
  typedef typename Superclass::MeasurementVectorType MeasurementVectorType;
  typedef typename Superclass::IndexArrayType        IndexArrayType;
  typedef typename Superclass::DistanceArrayType     DistanceArrayType;

  /** Typedefs from ANN. */
  typedef ANNpoint     ANNPointType;            // double *
  typedef ANNidx       ANNIndexType;            // int
  typedef ANNidxArray  ANNIndexArrayType;       // int *
  typedef ANNdist      ANNDistanceType;         // double
  typedef ANNdistArray ANNDistanceArrayType;    // double *

  /** An itk ANN tree. */
  typedef BinaryANNTreeBase< ListSampleType > BinaryANNTreeType;

  /** Set and get the binary tree. */
  virtual void SetBinaryTree( BinaryTreeType * tree );

  //const BinaryTreeType * GetBinaryTree( void ) const;

protected:

  BinaryANNTreeSearchBase();
  virtual ~BinaryANNTreeSearchBase();

  /** Member variables. */
  typename BinaryANNTreeType::Pointer m_BinaryTreeAsITKANNType;

private:

  BinaryANNTreeSearchBase( const Self & );  // purposely not implemented
  void operator=( const Self & );           // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryANNTreeSearchBase.hxx"
#endif

#endif // end #ifndef __itkBinaryANNTreeSearchBase_h
