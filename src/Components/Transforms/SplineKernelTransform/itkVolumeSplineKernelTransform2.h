/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkVolumeSplineKernelTransform2.h,v $
  Language:  C++
  Date:      $Date: 2006/03/18 18:06:38 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVolumeSplineKernelTransform2_h
#define __itkVolumeSplineKernelTransform2_h

#include "itkKernelTransform2.h"

namespace itk
{
/** \class VolumeSplineKernelTransform2
 * This class defines the thin plate spline (TPS) transformation.
 * It is implemented in as straightforward a manner as possible from
 * the IEEE TMI paper by Davis, Khotanzad, Flamig, and Harms,
 * Vol. 16 No. 3 June 1997
 *
 * \ingroup Transforms
 */
template< class TScalarType,         // Data type for scalars (float or double)
unsigned int NDimensions = 3 >
// Number of dimensions
class VolumeSplineKernelTransform2 :
  public KernelTransform2<   TScalarType, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef VolumeSplineKernelTransform2                 Self;
  typedef KernelTransform2< TScalarType, NDimensions > Superclass;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( VolumeSplineKernelTransform2, KernelTransform2 );

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType ParametersType;

  /** Jacobian Type */
  typedef typename Superclass::JacobianType JacobianType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass::SpaceDimension );

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
  typedef typename Superclass::InputPointType   InputPointType;
  typedef typename Superclass::OutputPointType  OutputPointType;
  typedef typename Superclass::InputVectorType  InputVectorType;
  typedef typename Superclass::OutputVectorType OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;
  typedef typename Superclass::PointsIterator PointsIterator;

protected:

  VolumeSplineKernelTransform2()
  {
    this->m_FastComputationPossible = true;
  }


  virtual ~VolumeSplineKernelTransform2() {}

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited. */
  typedef typename Superclass::GMatrixType GMatrixType;

  /** Compute G(x)
   * For the volume plate spline, this is:
   * G(x) = r(x)^3*I
   * \f$ G(x) = r(x)^3*I \f$
   * where
   * r(x) = Euclidean norm = sqrt[x1^2 + x2^2 + x3^2]
   * \f[ r(x) = \sqrt{ x_1^2 + x_2^2 + x_3^2 }  \f]
   * I = identity matrix. */
  void ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const;

  /** Compute the contribution of the landmarks weighted by the kernel funcion
      to the global deformation of the space  */
  virtual void ComputeDeformationContribution( const InputPointType & inputPoint,
    OutputPointType & result ) const;

private:

  VolumeSplineKernelTransform2( const Self & ); // purposely not implemented
  void operator=( const Self & );               // purposely not implemented

};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkVolumeSplineKernelTransform2.hxx"
#endif

#endif // __itkVolumeSplineKernelTransform2_h
