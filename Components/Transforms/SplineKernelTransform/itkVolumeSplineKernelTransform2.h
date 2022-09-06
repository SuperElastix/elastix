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
#ifndef itkVolumeSplineKernelTransform2_h
#define itkVolumeSplineKernelTransform2_h

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
template <class TScalarType, // Data type for scalars (float or double)
          unsigned int NDimensions = 3>
// Number of dimensions
class ITK_TEMPLATE_EXPORT VolumeSplineKernelTransform2 : public KernelTransform2<TScalarType, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VolumeSplineKernelTransform2);

  /** Standard class typedefs. */
  using Self = VolumeSplineKernelTransform2;
  using Superclass = KernelTransform2<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VolumeSplineKernelTransform2, KernelTransform2);

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
  VolumeSplineKernelTransform2() { this->m_FastComputationPossible = true; }


  ~VolumeSplineKernelTransform2() override = default;

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited. */
  using typename Superclass::GMatrixType;

  /** Compute G(x)
   * For the volume plate spline, this is:
   * G(x) = r(x)^3*I
   * \f$ G(x) = r(x)^3*I \f$
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
#  include "itkVolumeSplineKernelTransform2.hxx"
#endif

#endif // itkVolumeSplineKernelTransform2_h
