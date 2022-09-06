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
#ifndef itkANNBruteForceTree_h
#define itkANNBruteForceTree_h

#include "itkBinaryANNTreeBase.h"

namespace itk
{

/**
 * \class ANNBruteForceTree
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template <class TListSample>
class ITK_TEMPLATE_EXPORT ANNBruteForceTree : public BinaryANNTreeBase<TListSample>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ANNBruteForceTree);

  /** Standard itk. */
  using Self = ANNBruteForceTree;
  using Superclass = BinaryANNTreeBase<TListSample>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNBruteForceTree, BinaryANNTreeBase);

  /** Typedef's from Superclass. */
  using typename Superclass::SampleType;
  using typename Superclass::MeasurementVectorType;
  using typename Superclass::MeasurementVectorSizeType;
  using typename Superclass::TotalAbsoluteFrequencyType;

  /** Typedef's. */
  using ANNPointSetType = ANNpointSet;
  using ANNBruteForceTreeType = ANNbruteForce;

  /** Set the maximum number of points that are to be visited. */
  // void SetMaximumNumberOfPointsToVisit( unsigned int num )
  //{
  //  annMaxPtsVisit( static_cast<int>( num ) );
  //}

  /** Generate the tree. */
  void
  GenerateTree() override;

  /** Get the ANN tree. */
  ANNPointSetType *
  GetANNTree() const override
  {
    return this->m_ANNTree;
  }


protected:
  ANNBruteForceTree();
  ~ANNBruteForceTree() override;

  /** Member variables. */
  ANNBruteForceTreeType * m_ANNTree;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkANNBruteForceTree.hxx"
#endif

#endif // end #ifndef itkANNBruteForceTree_h
