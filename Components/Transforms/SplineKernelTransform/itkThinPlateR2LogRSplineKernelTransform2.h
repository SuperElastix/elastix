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
  Module:    $RCSfile: itkThinPlateR2LogRSplineKernelTransform2.h,v $
  Date:      $Date: 2006/03/19 04:36:59 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkThinPlateR2LogRSplineKernelTransform2_h
#define itkThinPlateR2LogRSplineKernelTransform2_h

#include "itkKernelTransform2.h"

namespace itk
{
/** \class ThinPlateR2LogRSplineKernelTransform2
 * This class defines the thin plate spline (TPS) transformation.
 * It is implemented in as straightforward a manner as possible from
 * the IEEE TMI paper by Davis, Khotanzad, Flamig, and Harms,
 * Vol. 16 No. 3 June 1997.
 *
 * The kernel used in this variant of TPS is \f$ R^2 log(R) \f$
 *
 * \ingroup Transforms
 */
template <typename TScalarType, // Data type for scalars (float or double)
          unsigned int NDimensions = 3>
// Number of dimensions
class ITK_TEMPLATE_EXPORT ThinPlateR2LogRSplineKernelTransform2 : public KernelTransform2<TScalarType, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ThinPlateR2LogRSplineKernelTransform2);

  /** Standard class typedefs. */
  using Self = ThinPlateR2LogRSplineKernelTransform2;
  using Superclass = KernelTransform2<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ThinPlateR2LogRSplineKernelTransform2);

  /** Scalar type. */
  using typename Superclass::ScalarType;

  /** Parameters type. */
  using typename Superclass::ParametersType;

  /** Jacobian Type */
  using typename Superclass::JacobianType;

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass::SpaceDimension);

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::PointsIterator;

protected:
  ThinPlateR2LogRSplineKernelTransform2() { this->m_FastComputationPossible = true; }


  ~ThinPlateR2LogRSplineKernelTransform2() override = default;

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited. */
  using typename Superclass::GMatrixType;

  /** Compute G(x)
   * For the thin plate spline, this is:
   * G(x) = r(x)^2 log(r(x)) * I
   * \f$ G(x) = r(x)^2 log(r(x)) * I \f$
   * where
   * r(x) = Euclidean norm = sqrt[x1^2 + x2^2 + x3^2]
   * \f[ r(x) = \sqrt{ x_1^2 + x_2^2 + x_3^2 }  \f]
   * I = identity matrix. */
  void
  ComputeG(const InputVectorType & x, GMatrixType & GMatrix) const override;

  /** Compute the contribution of the landmarks weighted by the kernel funcion
      to the global deformation of the space  */
  void
  ComputeDeformationContribution(const InputPointType & inputPoint, OutputPointType & result) const override;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkThinPlateR2LogRSplineKernelTransform2.hxx"
#endif

#endif // itkThinPlateR2LogRSplineKernelTransform2_h
