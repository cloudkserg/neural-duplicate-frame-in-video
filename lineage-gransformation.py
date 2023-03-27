from typing import List, Dict, Union

class LineageGraphTransformation:
    def __init__(self):
        self.groups: List[Dict[str, Union[str, bool]]] = []
        self.links: List[Dict[str, str]] = []
        self.nodes: List[Dict[str, Union[str, List[str]]]] = []

    def is_existing_group(self, area_key: str) -> bool:
        group_key = self.get_group_key(area_key)
        return any(group['name'] == group_key for group in self.groups)

    def add_transformation_group(self, group_name: str) -> None:
        if not self.is_existing_group(group_name):
            self.groups.insert(0, self.build_transformation_group(group_name))

    def build_transformation_group(self, group_name: str) -> Dict[str, Union[str, bool]]:
        return {
            'name': group_name,
            'header': group_name,
            'hidden': True
        }

    def get_links(self) -> List[Dict[str, str]]:
        return self.links

    def get_groups(self) -> List[Dict[str, Union[str, bool]]]:
        return self.groups

    def get_nodes(self) -> List[Dict[str, Union[str, List[str]]]]:
        return self.nodes

    def add_links_to_node(self, source_node_keys: List[str], next_node_key: str) -> None:
        for source_node_key in source_node_keys:
            link = self.build_link(source_node_key, next_node_key)

            if link and not self.is_existing_link(link):
                self.links.append(link)

    def is_existing_link(self, link: Dict[str, str]) -> bool:
        return any(compare_link['from'] == link['from'] and compare_link['to'] == link['to'] for compare_link in self.links)

    def build_link(self, source_node_key: str, next_node_key: str) -> Dict[str, str]:
        return {
            'from': source_node_key,
            'to': next_node_key
        }

    def add_transformation_node(self, node: Dict[str, Union[str, List[str]]]) -> None:
        self.add_transformation_group(node['groupName'])
        self.nodes.append(node)

    def add_transformations(self, transformation_node: Dict[str, Union[str, List[str], bool]]) -> None:
        if not transformation_node['isWithoutOperations']:
            self.add_transformations_with_operations(transformation_node)
        elif transformation_node['prevNodeKeys']:
            self.add_links_from_source_to_target_without_transformations(
                transformation_node['prevNodeKeys'],
                transformation_node['nextNodeKey']
            )

    def add_transformations_with_operations(self, transformation_node: Dict[str, Union[str, List[str], bool]]) -> None:
        self.add_transformation_node(transformation_node)
        self.add_links_from_source_to_target_through_transformation(transformation_node)

    def add_links_from_source_to_target_through_transformation(self, transformation_node: Dict[str, Union[str, List[str], bool]]) -> None:
        next_node_key = transformation_node['nextNodeKey']
        prev_node_keys = transformation_node['prevNodeKeys']
        self.add_links_to_node([transformation_node['key']], next_node_key)

        if prev_node_keys:
            self.add_links_to_node(prev_node_keys, transformation_node['key'])

    def add_links_from_source_to_target_without_transformations(self, prev_node_keys: List[str], next_node_key: str) -> None:
        if prev_node_keys:
            self.add_links_to_node(prev_node_keys,
