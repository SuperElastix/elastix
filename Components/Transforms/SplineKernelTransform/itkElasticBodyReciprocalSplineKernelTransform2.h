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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkElasticBodyReciprocalSplineKernelTransform2.h,v $
  Language:  C++
  Date:      $Date: 2006/04/17 01:50:19 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkElasticBodyReciprocalSplineKernelTransform2_h
#define itkElasticBodyReciprocalSplineKernelTransform2_h

#include "itkKernelTransform2.h"

namespace itk
{

/** \class ElasticBodyReciprocalSplineKernelTransform2
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
template <class TScalarType = double, // Data type for scalars (float or double)
          unsigned int NDimensions = 3>
// Number of dimensions
class ITK_TEMPLATE_EXPORT ElasticBodyReciprocalSplineKernelTransform2 : public KernelTransform2<TScalarType, NDimensions>
{
public:
  /** Standard class typedefs. */
  typedef ElasticBodyReciprocalSplineKernelTransform2 Self;
  typedef KernelTransform2<TScalarType, NDimensions>  Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElasticBodyReciprocalSplineKernelTransform2, KernelTransform2);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType ParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType JacobianType;

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass::SpaceDimension);

  /** Set alpha.  Alpha is related to Poisson's Ratio (\f$\nu\f$) as
   * \f$\alpha = 8 ( 1 - \nu ) - 1\f$
   */
  // itkSetMacro( Alpha, TScalarType ); Cant use the macro because the matrices must be recomputed
  void
  SetAlpha(TScalarType Alpha) override
  {
    this->m_Alpha = Alpha;
    this->m_LMatrixComputed = false;
    this->m_LInverseComputed = false;
    this->m_WMatrixComputed = false;
  }


  /** Get alpha */
  elxOverrideGetConstMacro(Alpha, TScalarType);

  /** Convenience method */
  void
  SetPoissonRatio(const TScalarType Nu) override
  {
    if (Nu > -1.0 && Nu < 0.5)
    {
      this->SetAlpha(8.0 * (1.0 - Nu) - 1.0);
    }
  }


  const TScalarType
  GetPoissonRatio(void) const override
  {
    return 1.0 - (this->m_Alpha + 1.0) / 8.0;
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
  ElasticBodyReciprocalSplineKernelTransform2();
  ~ElasticBodyReciprocalSplineKernelTransform2() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
  typedef typename Superclass::GMatrixType GMatrixType;

  /** Compute G(x)
   * For the elastic body spline, this is:
   * \f[ G(x) = [\alpha*r(x)*I - 3*x*x'/r(x) ] \f]
   * where
   * \f$\alpha = 8 ( 1 - \nu ) - 1\f$, \f$\nu\f$ is Poisson's Ratio,
   * \f$r(x) = \sqrt{ x_1^2 + x_2^2 + x_3^2 } \f$ and
   * \f$I\f$ is the identity matrix.
   */
  void
  ComputeG(const InputVectorType & x, GMatrixType & GMatrix) const override;

  /** alpha, Poisson's ratio */
  TScalarType m_Alpha;

private:
  ElasticBodyReciprocalSplineKernelTransform2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkElasticBodyReciprocalSplineKernelTransform2.hxx"
#endif

#endif // itkElasticBodyReciprocalSplineKernelTransform2_h
