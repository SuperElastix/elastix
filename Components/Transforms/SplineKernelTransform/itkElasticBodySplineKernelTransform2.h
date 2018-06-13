/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkElasticBodySplineKernelTransform2.h,v $
  Language:  C++
  Date:      $Date: 2003/09/10 14:29:06 $
  Version:   $Revision: 1.29 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkElasticBodySplineKernelTransform2_h
#define __itkElasticBodySplineKernelTransform2_h

#include "itkKernelTransform2.h"

namespace itk
{

/** \class ElasticBodySplineKernelTransform2
 * This class defines the elastic body spline (EBS) transformation.
 * It is implemented in as straightforward a manner as possible from
 * the IEEE TMI paper by Davis, Khotanzad, Flamig, and Harms,
 * Vol. 16 No. 3 June 1997
 * Taken from the paper:
 * The EBS "is based on a physical model of a homogeneous, isotropic,
 * three-dimensional elastic body. The model can approximate the way
 * that some physical objects deform".
 *
 * \ingroup Transforms
 */
template< class TScalarType = double,   // Data type for scalars (float or double)
unsigned int NDimensions    = 3 >
// Number of dimensions
class ElasticBodySplineKernelTransform2 :
  public KernelTransform2<  TScalarType, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef ElasticBodySplineKernelTransform2 Self;
  typedef KernelTransform2<  TScalarType,
    NDimensions > Superclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ElasticBodySplineKernelTransform2, KernelTransform2 );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType ParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType JacobianType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass::SpaceDimension );

  /** Set alpha.  Alpha is related to Poisson's Ratio (\f$\nu\f$) as
   * \f$\alpha = 12 ( 1 - \nu ) - 1\f$
   */
  //itkSetMacro( Alpha, TScalarType ); Cant use the macro because the matrices must be recomputed
  virtual void SetAlpha( TScalarType Alpha )
  {
    this->m_Alpha            = Alpha;
    this->m_LMatrixComputed  = false;
    this->m_LInverseComputed = false;
    this->m_WMatrixComputed  = false;
  }


  /** Get alpha */
  itkGetConstMacro( Alpha, TScalarType );

  /** Convenience method */
  virtual void SetPoissonRatio( const TScalarType Nu )
  {
    if( Nu > -1.0 && Nu < 0.5 )
    {
      this->SetAlpha( 12.0 * ( 1.0 - Nu ) - 1.0 );
    }
  }


  virtual const TScalarType GetPoissonRatio( void ) const
  {
    return 1.0 - ( this->m_Alpha + 1.0 ) / 12.0;
  }


  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;

protected:

  ElasticBodySplineKernelTransform2();
  virtual ~ElasticBodySplineKernelTransform2() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
  typedef typename Superclass::GMatrixType GMatrixType;

  /** Compute G(x)
   * For the elastic body spline, this is:
   * \f[ G(x) = [\alpha*r(x)^2*I - 3*x*x']*r(x) \f]
   * where
   * \f$\alpha = 12 ( 1 - \nu ) - 1\f$, \f$\nu\f$ is Poisson's Ratio,
   * \f$ r(x) = \sqrt{ x_1^2 + x_2^2 + x_3^2 } \f$ and
   * \f$I\f$ is the identity matrix.
   */
  void ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const;

  /** alpha,  Alpha is related to Poisson's Ratio \f$\nu\f$ as
   * \f$\alpha = 12 ( 1 - \nu ) - 1\f$
   */
  TScalarType m_Alpha;

private:

  ElasticBodySplineKernelTransform2( const Self & ); // purposely not implemented
  void operator=( const Self & );                    // purposely not implemented

};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkElasticBodySplineKernelTransform2.hxx"
#endif

#endif // __itkElasticBodySplineKernelTransform2_h
