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

#ifndef itkANNBinaryTreeCreator_h
#define itkANNBinaryTreeCreator_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "ANN/ANN.h"

namespace itk
{

/**
 * \class ANNBinaryTreeCreator
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

class ANNBinaryTreeCreator : public Object
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ANNBinaryTreeCreator);

  /** Standard itk. */
  using Self = ANNBinaryTreeCreator;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New method for creating an object using a factory. */
  itkNewMacro(Self);

  /** ITK type info. */
  itkTypeMacro(ANNBinaryTreeCreator, Object);

  /** ANN typedef's. */
  // typedef ANNpointSet     ANNTreeBaseType;
  using ANNkDTreeType = ANNkd_tree;
  using ANNbdTreeType = ANNbd_tree;
  using ANNBruteForceTreeType = ANNbruteForce;
  using ANNPointArrayType = ANNpointArray;
  using ANNSplitRuleType = ANNsplitRule;
  using ANNShrinkRuleType = ANNshrinkRule;

  /** Static funtions to create and delete ANN trees.
   * We keep a reference count so that when no more trees
   * of any sort exist, we can call annClose(). This little
   * function is cause of going through the trouble of creating
   * this class with static creating functions.
   */

  /** Static function to create an ANN kDTree. */
  static ANNkDTreeType *
  CreateANNkDTree(ANNPointArrayType pa, int n, int d, int bs = 1, ANNSplitRuleType split = ANN_KD_SUGGEST);

  /** Static function to create an ANN bdTree. */
  static ANNbdTreeType *
  CreateANNbdTree(ANNPointArrayType pa,
                  int               n,
                  int               d,
                  int               bs = 1,
                  ANNSplitRuleType  split = ANN_KD_SUGGEST,
                  ANNShrinkRuleType shrink = ANN_BD_SUGGEST);

  /** Static function to create an ANN BruteForceTree. */
  static ANNBruteForceTreeType *
  CreateANNBruteForceTree(ANNPointArrayType pa, int n, int d);

  /** Static function to delete any ANN tree that inherits from kDTree (not brute force). */
  static void
  DeleteANNkDTree(ANNkDTreeType *& tree);

  /** Static function to delete an ANN BruteForceTree. */
  static void
  DeleteANNBruteForceTree(ANNBruteForceTreeType *& tree);

  /** Static function to increase the reference count to ANN trees. */
  static void
  IncreaseReferenceCount();

  /** Static function to decrease the reference count to ANN trees. */
  static void
  DecreaseReferenceCount();

protected:
  ANNBinaryTreeCreator() = default;
  ~ANNBinaryTreeCreator() override = default;

private:
  /** Member variables. */
  static unsigned int m_NumberOfANNBinaryTrees;
};

} // end namespace itk

#endif // end #ifndef itkANNBinaryTreeCreator_h
